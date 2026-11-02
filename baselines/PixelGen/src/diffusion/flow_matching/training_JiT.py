# Modified from training_repa_JiT_LPIPS_DINO_NoiseGating.py
import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

def constant(alpha, sigma):
    return 1

def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


class JiTTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn: Callable=constant,
            lognorm_t=False,
            timeshift=1.0,
            P_mean=-0.8,
            P_std=0.8,
            t_eps=0.05,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = (torch.randn(batch_size, device=x.device, dtype=torch.float32)*self.P_std+self.P_mean).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        sigma = self.scheduler.sigma(t)

        x_t = alpha * x + noise * sigma
        v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        pred_img = net(x_t, t, y) if y_mask is None else net(x_t, t, y, y_mask=y_mask)
        out = (pred_img - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight * (out - v_t) ** 2

        return dict(
            fm_loss=fm_loss.mean(),
            loss=fm_loss.mean(),
            loss_per_sample=fm_loss.mean(dim=(1, 2, 3)),
        )