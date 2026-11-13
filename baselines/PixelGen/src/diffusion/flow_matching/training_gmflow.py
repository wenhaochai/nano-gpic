"""GMFlow's transition loss (Chen et al., ICML 2025; docs/gmflow.md, digest section 3) in our convention: t = 1 is data, sigma = 1 - t.

t is drawn as in JiTTrainer (logit-normal), so sigma_high = 1 - t; sigma_low = (1 - lam) sigma_high. x_low = (1 - sigma_low) x + sigma_low eps0, and x_high comes from x_low by
the forward transition. The network sees (x_high, t) and gives a per-pixel mixture over x; the loss scores x_low under the mixture mapped by the bridge:
    sum_k A_k N(c1 x_high + c2 mu_x,k, (c2^2 s_x^2 + c3) I),   s_x = sigma_high s_u,
as -logsumexp over K, summed over the 3 channels, averaged over the pixels, times 2 (GMFlow's weight_scale), averaged over the batch.
lam = 1 gives c1 = 0, c2 = 1, c3 = 0: the plain NLL of x under the x-mixture. There is no t_eps clamp: the NLL over x and over v differ by a constant in theta.
One mixture per image can anneal the posterior over its K candidates (anneal_tau0, anneal_steps; docs/gmflow.md section 16): tau = anneal_tau0 at step 0, geometric
down to 1 at step anneal_steps, then 1, the exact loss. The defaults (1, 0) keep the exact loss at every step.
"""
import os

import torch

from src.diffusion.flow_matching.gmflow import EPS, bridge_mixture, forward_transition, gm_component_ll, image_scope, transition_nll
from src.diffusion.flow_matching.training_JiT import JiTTrainer, time_shift_fn

SIGMA_BINS = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-6)


class GMFlowTrainer(JiTTrainer):
    def __init__(self, *args, lam=0.5, weight_scale=2.0, anneal_tau0=1.0, anneal_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 < lam <= 1, lam
        assert anneal_tau0 >= 1 and anneal_steps >= 0 and (anneal_tau0 == 1) == (anneal_steps == 0), (anneal_tau0, anneal_steps)
        self.lam, self.weight_scale = lam, weight_scale
        self.anneal_tau0, self.anneal_steps = float(anneal_tau0), int(anneal_steps)
        self.step = 0   # the global step, set by LightningModel.training_step (a resumed launch gets the checkpoint's step), for the temperature
        self.diag_now = False   # set by LightningModel.training_step on the steps whose logs are written: the diagnostics run only then, with no host sync
        self._print_every, self._calls = int(os.environ.get("GPIC_GM_PRINT") or 0), 0   # smoke test only: a host sync on these steps

    def tau(self, step):
        """The temperature of one mixture per image at a global step: anneal_tau0 ** (1 - step / anneal_steps) before anneal_steps, then exactly 1."""
        if step >= self.anneal_steps:
            return 1.0
        return self.anneal_tau0 ** (1 - step / self.anneal_steps)

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
            tau = self.tau(self.step)
            assert tau == 1 or image_scope(gm), "annealing tempers the posterior over whole candidate images: it needs gm_scope image"
            nll = transition_nll(gm, x_low, x_high, sigma_high, sigma_low, tau)       # (B, H, W); (B, 1, 1) for one mixture per image (section 15)
            per_sample = self.weight_scale * nll.mean((1, 2))
            out = dict(loss=per_sample.mean(), loss_per_sample=per_sample.detach())
            if self.diag_now:
                out.update(self.diagnostics(gm, sigma_high.view(-1)), **{"gm/nll": nll.detach().mean()})
                if image_scope(gm):
                    out.update(self.win_stats(gm, x_low, x_high, sigma_high, sigma_low, tau))
        if self._print_every:
            self._calls += 1
            if self._calls % self._print_every == 0:
                d = self.diagnostics(gm, sigma_high.view(-1))
                pix = f" logsu_pix_mean {d['gm/logsu_pix_mean'].item():.4f} logsu_pix_std {d['gm/logsu_pix_std'].item():.4f}" if "gm/logsu_pix_std" in d else ""
                if image_scope(gm):
                    ws = self.win_stats(gm, x_low, x_high, sigma_high, sigma_low, tau)
                    pix += f" win_max {ws['gm/win_max'].item():.3f} win_alive {int(ws['gm/win_alive'].item())} w_dead {ws['gm/w_dead'].item():.4f}"
                    if self.anneal_steps:
                        pix += f" step {self.step} tau {tau:.6g} r_neff {ws['gm/r_neff'].item():.3f}"
                    pix += f" mu_spread {ws['gm/mu_spread'].item():.4g} nll_exact {ws['gm/nll_exact'].item():.5f}"
                print(f"[gmflow] call {self._calls}: loss {out['loss'].item():.5f} s_u {d['gm/s_u'].item():.4f} n_eff {d['gm/n_eff'].item():.3f} max_w {d['gm/max_w'].item():.3f}{pix}", flush=True)
        return out

    @staticmethod
    @torch.no_grad()
    def win_stats(gm, x_low, x_high, sigma_high, sigma_low, tau=1.0):
        """One mixture per image (section 15): in this batch (one rank's images), the largest share of images that one component wins (gets the gradient of its mean),
        the number of components that win at least one image, and w_dead, the mean weight A on the components that win none: a component that never wins is never
        trained, and its weight still sends draws to its untrained candidate (review_image_mix S1). n_eff and max_w do not show this. one_hot, not bincount: bincount
        on CUDA syncs the host. The winners are those of the exact posterior at every tau: argmax of log A_k + ll_k, which the temperature does not change.
        Section 16 (review_image_anneal S1, S2): tau; r_neff, exp(entropy) of the responsibilities that weight the gradient at this tau, averaged over the images
        (1: one candidate gets it all; K: shared equally); nll_exact, the transition loss at tau = 1 on the scale of gm/nll (the logged loss is the tempered one);
        mu_spread, the RMS distance of the candidates from their mean in the transition's own std, per value (sqrt of mean_(K,C,H,W) (b_k - mean_k b_k)^2 / s^2 with
        b_k = c1 x_high + c2 mu_k, averaged over the images), and mu_spread_x, the same for the x-means in pixel units. Candidates that coincide give 0; a split shows
        as mu_spread rising while r_neff falls. One pass over the component log-likelihoods."""
        means, logstd = bridge_mixture(gm, x_high, sigma_high, sigma_low)                                 # (B, K, C, H, W), (B, 1, 1, 1, 1)
        z = gm_component_ll(x_low, means, logstd).sum((-2, -1)) + gm["logweights"].flatten(1)              # (B, K): log A_k + ll_k
        K, n_pix = z.shape[1], x_low.shape[-2] * x_low.shape[-1]
        share = torch.nn.functional.one_hot(z.argmax(1), K).float().mean(0)                              # (K,)
        A = gm["logweights"].exp().flatten(1).mean(0)                                                     # (K,)
        r = (z / tau).softmax(1)                                                                          # (B, K)
        r_neff = torch.exp(-(r * torch.log(r.clamp(min=1e-30))).sum(1)).mean()
        spread = lambda m: (m - m.mean(1, keepdim=True)).square().flatten(1).mean(1)                      # (B,): mean over (K, C, H, W)
        return {"gm/win_max": share.max(), "gm/win_alive": (share > 0).sum().float(), "gm/w_dead": (A * (share == 0)).sum(),
                "gm/tau": r.new_full((), tau), "gm/r_neff": r_neff,   # new_full: a fill kernel; torch.tensor(float, device) is a pageable copy (a host sync, perf C7)
                "gm/nll_exact": (-torch.logsumexp(z, dim=1) / n_pix).mean(),
                "gm/mu_spread": (spread(means) / torch.exp(2 * logstd).flatten()).sqrt().mean(), "gm/mu_spread_x": spread(gm["means"]).sqrt().mean()}

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
