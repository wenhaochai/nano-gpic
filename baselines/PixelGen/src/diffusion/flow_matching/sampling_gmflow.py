"""Samplers of the GMFlow head (docs/gmflow.md, digest sections 5 and 6), on the grid of the other samplers: t_i = i / N, so sigma_i = 1 - i / N, ending at sigma = 0.
num_steps is the number of network calls. The network is called as net(x, t, condition, gm_mode="full") and returns the per-pixel mixture over x (training_gmflow.py).
A point model (gm_k = 0) returns a tensor; it is read as K = 1 with s_x = 0. GM-SDE on it is DDPM with the small variance, the sampler control of the point model.
No CFG: the guidance is asserted 1.0.
"""
import os

import torch

from src.diffusion.flow_matching.gmflow import bridge_coefs, gm_convert_mean, gm_draw, gm_mean
from src.diffusion.flow_matching.sampling import EulerSamplerJiT


def as_mixture(out):
    """The network's output as a mixture in fp32. A tensor (a point model) becomes K = 1 with s_u = 0 (log s_u = -inf)."""
    if isinstance(out, dict):
        return {k: v.float() for k, v in out.items()}
    x = out.float()
    B = x.shape[0]
    return dict(means=x[:, None], logweights=torch.zeros(B, 1, 1, *x.shape[2:], device=x.device),
                logstd=torch.full((B, 1, 1, 1, 1), -float("inf"), device=x.device))


class _GMSampler(EulerSamplerJiT):
    def __init__(self, *args, rank_seed: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_seed = rank_seed   # off in the CPU tests only

    def _start(self, noise):
        assert float(self.guidance) == 1.0, "CFG is fixed at 1.0 (owner, 2026-11-01)"
        if self.rank_seed and not self._rank_seeded:   # seed_everything gives all ranks one RNG state, so their draws would repeat (as EulerSamplerJiT's eta path)
            torch.manual_seed(1000 + int(os.environ.get("LOCAL_RANK", 0)))
            self._rank_seeded = True
        return self.timesteps.to(noise.device, torch.float32)

    def _call(self, net, x, t_cur, condition):
        """A mixture (gm_k > 0) or, for a point model, its x prediction as an fp32 tensor. A point model is called without gm_mode."""
        t = t_cur.repeat(x.shape[0])
        if getattr(self, "_point", None) is None:
            out = net(x, t, condition, gm_mode="full")
            self._point = not isinstance(out, dict)
            return out.float() if self._point else as_mixture(out)
        return net(x, t, condition).float() if self._point else as_mixture(net(x, t, condition, gm_mode="full"))


class GMSDESampler(_GMSampler):
    """GM-SDE, first order: x0_hat ~ q(x0 | x_t), independently per pixel, then the bridge step x' = c1 x + c2 x0_hat + sqrt(c3) xi. The last step (c1 = 0, c2 = 1, c3 = 0)
    returns the draw. final_mean: the last step uses the mixture mean instead of a draw. No spectral sampling, no second order."""
    def __init__(self, *args, final_mean: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_mean = final_mean

    def _impl_sampling(self, net, noise, condition, uncondition):
        steps = self._start(noise)
        dev = noise.device.type
        x = noise.float()
        x_trajs, v_trajs = [x], []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            gm = self._call(net, x, t_cur, condition)
            with torch.autocast(device_type=dev, enabled=False):
                sigma, sigma_to = (1 - t_cur).view(1, 1, 1, 1), (1 - t_next).view(1, 1, 1, 1)
                last = i == len(steps) - 2
                if torch.is_tensor(gm):   # a point model: K = 1, s_x = 0, so the draw is its x prediction (DDPM with the small variance)
                    x0 = gm
                else:
                    x0 = gm_mean(gm) if (last and self.final_mean) else gm_draw(gm, sigma.expand(x.shape[0], -1, -1, -1))
                c1, c2, c3 = bridge_coefs(sigma, sigma_to)
                x_new = c1 * x + c2 * x0 + c3.clamp(min=0).sqrt() * torch.randn_like(x)
            v_trajs.append((x0 - x) / sigma)
            x = x_new
            x_trajs.append(x)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs


class GMODESampler(_GMSampler):
    """GM-ODE, first order, with `substeps` analytic sub-steps per call (GMFlow gmflow.py:527-539). Sub-step 0 uses the mixture mean; sub-step j > 0 re-weights the call's
    mixture to the current state (Eq. 10, gm_convert) and uses its mean. Each sub-step is the Euler step x' = x + (sigma' - sigma) (x - x0) / sigma, on a uniform sigma grid of
    num_steps * substeps points. The last sub-step (sigma' = 0) returns the re-weighted mixture mean."""
    def __init__(self, *args, substeps: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        assert substeps >= 1, substeps
        self.substeps = substeps

    def _impl_sampling(self, net, noise, condition, uncondition):
        steps = self._start(noise)
        dev = noise.device.type
        x = noise.float()
        x_trajs, v_trajs = [x], []
        for t_cur, t_next in zip(steps[:-1], steps[1:]):
            gm = self._call(net, x, t_cur, condition)
            gm = as_mixture(gm) if torch.is_tensor(gm) else gm
            with torch.autocast(device_type=dev, enabled=False):
                sigma_src, sigma_end = (1 - t_cur).view(1, 1, 1, 1), (1 - t_next).view(1, 1, 1, 1)
                sig_src_b = sigma_src.expand(x.shape[0], -1, -1, -1)
                x_src = x
                for j in range(self.substeps):
                    s = sigma_src + (sigma_end - sigma_src) * (j / self.substeps)
                    s_next = sigma_src + (sigma_end - sigma_src) * ((j + 1) / self.substeps)
                    x0 = gm_mean(gm) if j == 0 else gm_convert_mean(gm, sig_src_b, s.expand(x.shape[0], -1, -1, -1), x_src, x)
                    x = x + (s_next - s) * (x - x0) / s
            v_trajs.append((x - x_src) / (sigma_src - sigma_end))
            x_trajs.append(x)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs
