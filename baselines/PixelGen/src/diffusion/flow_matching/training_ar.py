"""The autoregressive model's loss (docs/autoregressive.md): teacher forcing, as an LLM trains. One forward gives z at every position from the true patches before it, and the head trains on each position's patch. Each position has its own head time tau and noise.

The head's recipe is the official one, per patch: x-prediction, L2 in v space with t_eps, except for a wider tau: logit-normal(-0.8, 1.6). In tools/ar/head_modes.py (8 modes in 768 dimensions, one training seed) a draw lies closer to its mode: rms 0.005 against 0.012 at the official std of 0.8. The modes' shares do not separate the two: total variation 0.027 against 0.040, with a sampling noise of about 0.016. CFG dropout comes from BaseTrainer.
"""
import torch

from src.diffusion.base.training import BaseTrainer


class ARTrainer(BaseTrainer):
    def __init__(self, P_mean=-0.8, P_std=1.6, t_eps=0.05, **kwargs):
        super().__init__(**kwargs)
        self.P_mean, self.P_std, self.t_eps = P_mean, P_std, t_eps

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        patches = torch.nn.functional.unfold(x, kernel_size=net.patch_size, stride=net.patch_size).transpose(1, 2)
        B, N, _ = patches.shape
        tau = (torch.randn(B, N, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()
        eta = torch.randn_like(patches)
        q = tau[..., None] * patches + (1 - tau[..., None]) * eta
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        context = patches + net.context_noise * torch.randn_like(patches) if net.context_noise else patches   # what the backbone reads; the target stays clean
        hat = net(context, y, y_mask=y_mask, q=q, tau=tau).float()
        # v_hat - v = ((hat - q) - (patches - q)) / (1 - tau), with the official clamp
        per_value = ((hat - patches) / (1 - tau[..., None]).clamp_min(self.t_eps)) ** 2
        fm = per_value.mean()
        return dict(fm_loss=fm, loss=fm, loss_per_sample=per_value.mean(dim=(1, 2)))
