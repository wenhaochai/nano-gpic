"""Implicit sampler: the network draws samples of p(x | x_t, y) instead of giving their mean (docs/distribution_step.md).

The loss is a generalized energy score over whole images, on the same v-space scale as the JiT recipe:
    E ||w (g - x)||^beta - 1/2 E ||w (g - g')||^beta,   w = 1 / max(1 - t, t_eps),   beta in (0, 2)
with g, g' two samples for one x_t. It is strictly proper, so its minimum is the true posterior, and it needs only one true image per x_t.
The norm is the root mean square over pixels, which only rescales the score. Everything else is JiTTrainer's: t, noise, the CFG dropout.

ddm (2026-11-22, docs/ddm.md; iDDM, arXiv 2609.37147) adds two options; their defaults give the trainer above, unchanged:
- kernel_patch p > 0: iDDM's local kernel (their App. F, eq. 28): the score on each p x p patch of all C channels (one token of the
  model at patch size p), averaged over patches, in place of one score over the whole image. Each patch's norm is its root mean square
  (the whole-image score above uses the same norm), so the score's scale does not grow with the patch's 3 p^2 values; iDDM's latent
  tokens hold 16 or 128 values, a pixel token 768.
- schedule "linear": iDDM's default time schedules (their eq. 7, linear profile, s(t) = 1 - t in this file's t, 1 = clean):
  lambda(t) = lam (1 - t), beta(t) = 2 - (2 - beta) (1 - t). So beta is beta(0) (iDDM: 0.1) and lam is lambda(0) (iDDM: 1); near the
  clean end the score turns into the squared error of each sample. "const" = the fixed lam and beta above.
"""
import os

import torch

from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn


class ImplicitJiTTrainer(JiTTrainer):
    def __init__(self, *args, samples=2, beta=1.0, lam=1.0, low_t_p=0.0, low_t_max=0.05, kernel_patch=0, schedule="const", **kwargs):
        super().__init__(*args, **kwargs)
        assert samples >= 2 and 0 < beta < 2, (samples, beta)
        assert kernel_patch >= 0 and schedule in ("const", "linear"), (kernel_patch, schedule)
        self.samples, self.beta, self.lam = samples, beta, lam   # lam = 1 is strictly proper; De Bortoli et al. (arXiv 2502.02483) find lam < 1 better at many steps
        self.kernel_patch, self.schedule = kernel_patch, schedule
        self.low_t_p, self.low_t_max = low_t_p, low_t_max   # share of images with t uniform below low_t_max, where a small set has a posterior over many images
        self._print_every, self._calls = int(os.environ.get("GPIC_IMPLICIT_PRINT") or 0), 0   # smoke test only: a host sync on these steps

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        B = x.shape[0]
        if self.lognorm_t:
            base_t = (torch.randn(B, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()
        else:
            base_t = torch.rand(B, device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        if self.low_t_p:
            t = torch.where(torch.rand(B, device=x.device) < self.low_t_p, torch.rand(B, device=x.device) * self.low_t_max, t)
        noise = torch.randn_like(x)
        x_t = self.scheduler.alpha(t) * x + noise * self.scheduler.sigma(t)
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        g = net(x_t, t, y, m=self.samples) if y_mask is None else net(x_t, t, y, y_mask=y_mask, m=self.samples)
        g = g.view(B, self.samples, *x.shape[1:]).float()
        score, to_data, between = self.score(g, x, t)
        if self._print_every:
            self._calls += 1
            if self._calls % self._print_every == 0:
                print(f"[implicit] call {self._calls}: score {score.mean().item():.5f} to_data {to_data.mean().item():.5f} between {between.mean().item():.5f}", flush=True)
        return dict(loss=score.mean(), to_data=to_data.detach().mean(), between=between.detach().mean(), loss_per_sample=score.detach())

    def score(self, g, x, t):
        """g: [B, samples, C, H, W] fp32 draws for x [B, C, H, W] at t [B]; returns the per-image score, its term to the data and its term between
        the samples, each [B], in v space."""
        w = 1 / (1 - t.view(-1, 1, 1, 1, 1)).clamp_min(self.t_eps)
        if self.schedule == "linear":   # per image: [B, 1], against the [B, samples] or [B, pairs] norms below
            s_t = (1 - t).view(-1, 1)
            half, lam = (2 - (2 - self.beta) * s_t) / 2, self.lam * s_t.view(-1)
        else:
            half, lam = self.beta / 2, self.lam
        i, j = torch.triu_indices(g.shape[1], g.shape[1], 1, device=g.device)
        to_data = self._norm_pow(w * (g - x[:, None]), half).mean(1)              # [B]
        between = self._norm_pow(w * (g[:, i] - g[:, j]), half).mean(1)           # [B]
        return to_data - 0.5 * lam * between, to_data, between

    def _norm_pow(self, d, half):
        """d: [B, n, C, H, W] differences; returns [B, n]: (mean square + 1e-8)^half, over the whole image (kernel_patch 0) or per
        kernel_patch x kernel_patch patch and then averaged over patches. half: a number or a [B, 1] tensor."""
        p = self.kernel_patch
        if not p:
            return (d ** 2).mean((2, 3, 4)).add(1e-8).pow(half)
        B, n, C, H, W = d.shape
        ms = (d ** 2).view(B, n, C, H // p, p, W // p, p).mean((2, 4, 6))                    # [B, n, H / p, W / p]
        return ms.add(1e-8).pow(half if not torch.is_tensor(half) else half.view(B, 1, 1, 1)).mean((2, 3))
