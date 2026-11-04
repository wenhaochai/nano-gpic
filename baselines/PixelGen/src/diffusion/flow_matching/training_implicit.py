"""Implicit sampler: the network draws samples of p(x | x_t, y) instead of giving their mean (docs/distribution_step.md).

The loss is a generalized energy score over whole images, on the same v-space scale as the JiT recipe:
    E ||w (g - x)||^beta - 1/2 E ||w (g - g')||^beta,   w = 1 / max(1 - t, t_eps),   beta in (0, 2)
with g, g' two samples for one x_t. It is strictly proper, so its minimum is the true posterior, and it needs only one true image per x_t.
The norm is the root mean square over pixels, which only rescales the score. Everything else is JiTTrainer's: t, noise, the CFG dropout.
"""
import os

import torch

from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn


class ImplicitJiTTrainer(JiTTrainer):
    def __init__(self, *args, samples=2, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert samples >= 2 and 0 < beta < 2, (samples, beta)
        self.samples, self.beta = samples, beta
        self._print_every, self._calls = int(os.environ.get("GPIC_IMPLICIT_PRINT") or 0), 0   # smoke test only: a host sync on these steps

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        B = x.shape[0]
        if self.lognorm_t:
            base_t = (torch.randn(B, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()
        else:
            base_t = torch.rand(B, device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        noise = torch.randn_like(x)
        x_t = self.scheduler.alpha(t) * x + noise * self.scheduler.sigma(t)
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        g = net(x_t, t, y, m=self.samples) if y_mask is None else net(x_t, t, y, y_mask=y_mask, m=self.samples)
        g = g.view(B, self.samples, *x.shape[1:]).float()
        w = 1 / (1 - t.view(-1, 1, 1, 1, 1)).clamp_min(self.t_eps)
        half = self.beta / 2
        to_data = ((w * (g - x[:, None])) ** 2).mean((2, 3, 4)).add(1e-8).pow(half).mean(1)               # [B]
        i, j = torch.triu_indices(self.samples, self.samples, 1, device=x.device)
        between = ((w * (g[:, i] - g[:, j])) ** 2).mean((2, 3, 4)).add(1e-8).pow(half).mean(1)            # [B]
        score = to_data - 0.5 * between
        if self._print_every:
            self._calls += 1
            if self._calls % self._print_every == 0:
                print(f"[implicit] call {self._calls}: score {score.mean().item():.5f} to_data {to_data.mean().item():.5f} between {between.mean().item():.5f}", flush=True)
        return dict(loss=score.mean(), to_data=to_data.detach().mean(), between=between.detach().mean(), loss_per_sample=score.detach())
