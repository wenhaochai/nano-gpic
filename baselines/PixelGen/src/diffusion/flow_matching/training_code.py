"""Plan A (docs/distribution_step.md 5.2): the point recipe, conditioned on a code z of the image. The network gives a family of conditional means, one for each z, and that family
stands for p(x | x_t, y). With code_dim = 0 this is JiTTrainer's loss on the same RNG stream.

Everything else is JiTTrainer's: t, noise, the CFG dropout, the v-space L2. The encoder needs no KL term: its channel has a fixed capacity (JiT_T2I.code_encode).
The flow on z trains with z detached, so it cannot change the code. Its loss has a small weight: AdamW normalizes each parameter's gradient, so the flow learns at the same speed, and the
global gradient clip keeps acting on the main loss alone.
"""
import os

import torch

from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn


class CodeJiTTrainer(JiTTrainer):
    def __init__(self, *args, prior_weight=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_weight = prior_weight
        self._check_every, self._calls = int(os.environ.get("GPIC_CODE_CHECK") or 2000), 0   # a host sync on these steps

    def _check(self, net, z, y, y_mask, fm, prior):
        """Tells a bad code from a bad flow. z_var is near 1 - a^2 if the code collapsed (the encoder gives one direction for all images) and near 1 if the code is spread.
        flow_ed is the energy distance between the batch's z and samples of the flow, on the same captions."""
        with torch.no_grad():
            flow = net.code_flow(torch.randn_like(z), y, y_mask)
            d = lambda a, b: torch.cdist(a, b).sum() / (a.shape[0] * (b.shape[0] - (a is b)))
            ed = 2 * d(z, flow) - d(z, z) - d(flow, flow)
            print(f"[code] call {self._calls}: fm {fm.item():.5f} prior {prior.item():.5f} z_var {z.var(0).mean().item():.4f} flow_ed {ed.item():.5f}", flush=True)

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
                self._check(net, z.detach(), y, y_mask, fm, prior)
        return res
