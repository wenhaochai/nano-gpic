"""GMFlow's transition loss (Chen et al., ICML 2025; docs/gmflow.md, digest section 3) in our convention: t = 1 is data, sigma = 1 - t.

t is drawn as in JiTTrainer (logit-normal), so sigma_high = 1 - t; sigma_low = (1 - lam) sigma_high. x_low = (1 - sigma_low) x + sigma_low eps0, and x_high comes from x_low by
the forward transition. The network sees (x_high, t) and gives a per-pixel mixture over x; the loss scores x_low under the mixture mapped by the bridge:
    sum_k A_k N(c1 x_high + c2 mu_x,k, (c2^2 s_x^2 + c3) I),   s_x = sigma_high s_u,
as -logsumexp over K, summed over the 3 channels, averaged over the pixels, times 2 (GMFlow's weight_scale), averaged over the batch.
lam = 1 gives c1 = 0, c2 = 1, c3 = 0: the plain NLL of x under the x-mixture. There is no t_eps clamp: the NLL over x and over v differ by a constant in theta.
"""
import os

import torch

from src.diffusion.flow_matching.gmflow import EPS, forward_transition, transition_nll
from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn

SIGMA_BINS = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-6)


class GMFlowTrainer(JiTTrainer):
    def __init__(self, *args, lam=0.5, weight_scale=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 < lam <= 1, lam
        self.lam, self.weight_scale = lam, weight_scale
        self.diag_now = False   # set by LightningModel.training_step on the steps whose logs are written: the diagnostics run only then, with no host sync
        self._print_every, self._calls = int(os.environ.get("GPIC_GM_PRINT") or 0), 0   # smoke test only: a host sync on these steps

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        B = x.shape[0]
        if self.lognorm_t:
            base_t = (torch.randn(B, device=x.device, dtype=torch.float32) * self.P_std + self.P_mean).sigmoid()
        else:
            base_t = torch.rand(B, device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            sigma_high = (1 - t).clamp(min=EPS, max=1.0).view(-1, 1, 1, 1)            # GMFlow gmflow.py:392
            sigma_low = sigma_high * (1 - self.lam)   # GMFlow also takes min(., sigma_high - 1e-4 / 1000); with lam = 0.5 that only acts at sigma_high < 2e-7, below our clamp
            noise = torch.randn_like(xf)
            x_low = (1 - sigma_low) * xf + sigma_low * noise
            x_high = forward_transition(x_low, sigma_low, sigma_high, torch.randn_like(xf))
        t_net = 1 - sigma_high.view(-1)
        y_mask = metadata.get("y_mask") if isinstance(metadata, dict) else None
        gm = net(x_high, t_net, y, gm_mode="full") if y_mask is None else net(x_high, t_net, y, y_mask=y_mask, gm_mode="full")
        with torch.autocast(device_type=x.device.type, enabled=False):
            gm = {k: v.float() for k, v in gm.items()}
            nll = transition_nll(gm, x_low, x_high, sigma_high, sigma_low)            # (B, H, W)
            per_sample = self.weight_scale * nll.mean((1, 2))
            out = dict(loss=per_sample.mean(), loss_per_sample=per_sample.detach())
            if self.diag_now:
                out.update(self.diagnostics(gm, sigma_high.view(-1)), **{"gm/nll": nll.detach().mean()})
        if self._print_every:
            self._calls += 1
            if self._calls % self._print_every == 0:
                d = self.diagnostics(gm, sigma_high.view(-1))
                pix = f" logsu_pix_mean {d['gm/logsu_pix_mean'].item():.4f} logsu_pix_std {d['gm/logsu_pix_std'].item():.4f}" if "gm/logsu_pix_std" in d else ""
                print(f"[gmflow] call {self._calls}: loss {out['loss'].item():.5f} s_u {d['gm/s_u'].item():.4f} n_eff {d['gm/n_eff'].item():.3f} max_w {d['gm/max_w'].item():.3f}{pix}", flush=True)
        return out

    @staticmethod
    @torch.no_grad()
    def diagnostics(gm, sigma):
        """Mean s_u; per-pixel effective number of components exp(H(A)); the largest of the K mean weights; s_u and n_eff in four bins of sigma_high (NaN when a bin is empty).
        With a per-pixel spread (logstd (B, 1, 1, H, W)): s_u is each sample's mean over its pixels, and gm/logsu_pix_mean and gm/logsu_pix_std are the mean and the
        standard deviation over the pixels of log s_u, averaged over the batch."""
        lw = gm["logweights"].squeeze(2)                                   # (B, K, H, W)
        w = lw.exp()
        n_eff = torch.exp(-(w * lw).sum(1)).mean((1, 2))                   # (B,)
        ls = gm["logstd"]
        s_u = ls.exp().flatten(1).mean(1)                                  # (B,); one value per sample unless the spread is per pixel
        out = {"gm/s_u": s_u.mean(), "gm/n_eff": n_eff.mean(), "gm/max_w": w.mean((0, 2, 3)).max()}
        if ls[0].numel() > 1:
            out["gm/logsu_pix_mean"] = ls.flatten(1).mean(1).mean()
            out["gm/logsu_pix_std"] = ls.flatten(1).std(1).mean()
        for lo, hi in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
            m = ((sigma >= lo) & (sigma < hi)).float()
            n = m.sum()
            tag = f"{lo:.2f}-{min(hi, 1.0):.2f}"
            out[f"gm/s_u_sigma{tag}"] = (s_u * m).sum() / n                # 0 / 0 = NaN when empty
            out[f"gm/n_eff_sigma{tag}"] = (n_eff * m).sum() / n
        return out
