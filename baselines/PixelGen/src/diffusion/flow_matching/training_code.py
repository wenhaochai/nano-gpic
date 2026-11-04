"""Plan A (docs/distribution_step.md 5.2): the point recipe, conditioned on a code z of the image. The network gives a family of conditional means, one for each z, and that family
stands for p(x | x_t, y). With code_dim = 0 this is JiTTrainer's loss on the same RNG stream.

Everything else is JiTTrainer's: t, noise, the CFG dropout, the v-space L2. The encoder needs no KL term: its channel has a fixed capacity (JiT_T2I.code_encode).
The flow on z trains with z detached, so it cannot change the code. Its loss has a small weight: AdamW normalizes each parameter's gradient, so the flow learns at the same speed, and the
weight keeps the flow's share of the global gradient clip small (lightning_model_gpic.py prints both norms at each check).
"""
import os

import torch

from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn


class CodeJiTTrainer(JiTTrainer):
    def __init__(self, *args, prior_weight=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_weight = prior_weight
        self._check_every, self._calls = int(os.environ.get("GPIC_CODE_CHECK") or 2000), 0   # a host sync on these steps

    def _check(self, net, ema_net, x, z, y, y_mask, fm, prior):
        """Tells a bad code from a bad flow. z_var is near 1 - a^2 if the code collapsed (the encoder gives one direction for all images) and near 1 if the code is spread.
        flow_ed is the energy distance between the batch's z and samples of the flow, on the same captions. ema_flow_ed is the same for the EMA encoder and flow: sampling uses the EMA flow and decoder,
        so a code that rotates within the EMA window shows here. proj_std is the size of z in the conditioning, on the scale of the timestep embedding (std 1.44 in the trained baseline)."""
        d = lambda a, b: torch.cdist(a, b).sum() / (a.shape[0] * (b.shape[0] - (a is b)))
        ed = lambda a, b: (2 * d(a, b) - d(a, a) - d(b, b)).item()
        with torch.no_grad():
            flow_ed = ed(z, net.code_flow(torch.randn_like(z), y, y_mask))
            text = ""
            if ema_net is not None:
                text = f" ema_flow_ed {ed(ema_net.code_encode(x), ema_net.code_flow(torch.randn_like(z), y, y_mask)):.5f}"
            print(f"[code] call {self._calls}: fm {fm.item():.5f} prior {prior.item():.5f} z_var {z.var(0).mean().item():.4f} proj_std {net.code_proj(z).std().item():.4f} flow_ed {flow_ed:.5f}{text}", flush=True)

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        B = x.shape[0]
        if self.lognorm_t:
            base_t = (torch.randn(B, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()
        else:
            base_t = torch.rand(B, device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        noise = torch.randn_like(x)
        alpha, sigma = self.scheduler.alpha(t), self.scheduler.sigma(t)
        x_t = alpha * x + noise * sigma
        v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        kw = {} if y_mask is None else {"y_mask": y_mask}
        z = net.code_encode(x) if net.code_dim else None
        if z is not None:
            kw["z"] = z
        pred = net(x_t, t, y, **kw)
        out = (pred - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)
        fm_loss = self.loss_weight_fn(alpha, sigma) * (out - v_t) ** 2
        fm = fm_loss.mean()
        res = dict(fm_loss=fm, loss=fm, loss_per_sample=fm_loss.mean(dim=(1, 2, 3)))
        if z is not None:
            prior = net.code_prior_loss(z.detach(), y, y_mask)
            res["prior_loss"] = prior.detach()
            res["loss"] = fm + self.prior_weight * prior
            self._calls += 1
            if self._calls % self._check_every == 0:
                self._check(net, ema_net, x, z.detach(), y, y_mask, fm, prior)
        return res
