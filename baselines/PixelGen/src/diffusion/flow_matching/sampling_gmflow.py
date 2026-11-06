"""Samplers of the GMFlow head (docs/gmflow.md, digest sections 5 and 6), on the grid of the other samplers: t_i = i / N, so sigma_i = 1 - i / N, ending at sigma = 0.
num_steps is the number of network calls. The network is called as net(x, t, condition, gm_mode="full") and returns the per-pixel mixture over x (training_gmflow.py).
A point model (gm_k = 0) returns a tensor; it is read as K = 1 with s_x = 0. GM-SDE on it is DDPM with the small variance, the sampler control of the point model.
No CFG: the guidance is asserted 1.0.
order 2 (GM-SDE 2, GM-ODE 2; paper App. A.2, GMFlow gmflow.py:251-291 and 497-503): from the second call on, the call's mixture is reweighted by gm_second_order with the
previous call's raw mixture converted to the current x_t (Eq. 10); then the first-order step or the sub-steps run on it. The first call is first order. order 1 is the
code before this option, unchanged. On a point model (s = 0, K = 1) order 2 changes nothing (its surrogate variance is 0); GMODESampler refuses order 2 with a
spread table, whose surrogate variance s_x^2 > 0 would make a new, untested sampler.
"""
import os

import torch

from src.diffusion.flow_matching.gmflow import bridge_coefs, gm_convert_mean, gm_draw, gm_mean, gm_second_order
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
    def __init__(self, *args, rank_seed: bool = True, order: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert order in (1, 2), order
        self.rank_seed = rank_seed   # off in the CPU tests only
        self.order = order

    def _start(self, noise):
        assert float(self.guidance) == 1.0, "CFG is fixed at 1.0 (owner, 2026-11-01)"
        if self.rank_seed and not self._rank_seeded:   # seed_everything gives all ranks one RNG state, so their draws would repeat (as EulerSamplerJiT's eta path)
            torch.manual_seed(1000 + int(os.environ.get("LOCAL_RANK", 0)))
            self._rank_seeded = True
        return self.timesteps.to(noise.device, torch.float32)

    def _call(self, net, x, t_cur, condition):
        """A mixture (gm_k > 0) or, for a point model, its x prediction as an fp32 tensor. JiT_T2I with gm_k = 0 ignores gm_mode, so every call has one signature."""
        out = net(x, t_cur.repeat(x.shape[0]), condition, gm_mode="full")
        return as_mixture(out) if isinstance(out, dict) else out.float()

    def _second_order(self, gm, x, sigma, sigma_to, prev):
        """order 2: (the mixture to step with, the state for the next call). prev = (raw mixture, x, sigma, h) of the previous call, None at the first call (first order,
        GMFlow gmflow.py:312-313). h = sigma - sigma_to, the call's step; the last call's h is sigma (to 0), as the code. sigma (1, 1, 1, 1)."""
        h = sigma - sigma_to
        state = (gm, x, sigma, h)
        if prev is None:
            return gm, state
        p_gm, p_x, p_sigma, p_h = prev
        B = x.shape[0]
        mean_prev = gm_convert_mean(p_gm, p_sigma.expand(B, -1, -1, -1), sigma.expand(B, -1, -1, -1), p_x, x)
        return gm_second_order(gm, mean_prev, sigma.expand(B, -1, -1, -1), h, p_h), state


class GMSDESampler(_GMSampler):
    """GM-SDE, first order: x0_hat ~ q(x0 | x_t), independently per pixel, then the bridge step x' = c1 x + c2 x0_hat + sqrt(c3) xi. The last step (c1 = 0, c2 = 1, c3 = 0)
    returns the draw. final_mean: the last step uses the mixture mean instead of a draw. No spectral sampling. order 2: GM-SDE 2 (module docstring)."""
    def __init__(self, *args, final_mean: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_mean = final_mean

    def _impl_sampling(self, net, noise, condition, uncondition):
        steps = self._start(noise)
        dev = noise.device.type
        x = noise.float()
        x_trajs, v_trajs = [x], []
        prev = None
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            gm = self._call(net, x, t_cur, condition)
            with torch.autocast(device_type=dev, enabled=False):
                sigma, sigma_to = (1 - t_cur).view(1, 1, 1, 1), (1 - t_next).view(1, 1, 1, 1)
                if self.order == 2 and not torch.is_tensor(gm):
                    gm, prev = self._second_order(gm, x, sigma, sigma_to, prev)
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
    num_steps * substeps points. The last sub-step (sigma' = 0) returns the re-weighted mixture mean. order 2: GM-ODE 2 (module docstring), the sub-steps on the
    second-order mixture."""
    def __init__(self, *args, substeps: int = 3, spread_table: str = None, fused: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        assert substeps >= 1, substeps
        self.substeps = substeps
        # docs/gmflow_hn.md: spread_table (a point model only) = a JSON of s_x(t) (tools/gmflow/calibrate_spread.py); fused = the Triton kernel for the sub-steps (gm_ode_triton.py)
        self.spread = None if spread_table is None else __import__("src.diffusion.flow_matching.gm_spread", fromlist=["SpreadTable"]).SpreadTable(spread_table)
        self.fused = fused
        self._grid_w = {}   # (substeps, device) -> the fp32 weights j / n of the fused path's sigma grid
        assert not (self.spread is not None and self.order == 2), "a spread table with order 2 is an untested sampler (review_gmflow_hn F7)"

    def _impl_sampling(self, net, noise, condition, uncondition):
        steps = self._start(noise)
        dev = noise.device.type
        x = noise.float()
        x_trajs, v_trajs = [x], []
        prev = None
        for t_cur, t_next in zip(steps[:-1], steps[1:]):
            gm = self._call(net, x, t_cur, condition)
            assert self.spread is None or torch.is_tensor(gm), "spread_table is for a point model (gm_k = 0)"
            gm = self._point_gm(gm, t_cur) if torch.is_tensor(gm) else gm
            with torch.autocast(device_type=dev, enabled=False):
                sigma_src, sigma_end = (1 - t_cur).view(1, 1, 1, 1), (1 - t_next).view(1, 1, 1, 1)
                if self.order == 2:
                    gm, prev = self._second_order(gm, x, sigma_src, sigma_end, prev)
                sig_src_b = sigma_src.expand(x.shape[0], -1, -1, -1)
                x_src = x
                if self.fused:
                    x = self._fused_substeps(gm, sigma_src, sigma_end, sig_src_b, x)
                for j in range(0 if self.fused else self.substeps):
                    s = sigma_src + (sigma_end - sigma_src) * (j / self.substeps)
                    s_next = sigma_src + (sigma_end - sigma_src) * ((j + 1) / self.substeps)
                    x0 = gm_mean(gm) if j == 0 else gm_convert_mean(gm, sig_src_b, s.expand(x.shape[0], -1, -1, -1), x_src, x)
                    x = x + (s_next - s) * (x - x0) / s
            v_trajs.append((x - x_src) / (sigma_src - sigma_end))
            x_trajs.append(x)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs

    def _point_gm(self, x_hat, t_cur):
        """A point model's output as a mixture: K = 1 with s_x = 0 (as_mixture), or with s_x = the spread table at the call's t (docs/gmflow_hn.md, H2)."""
        if self.spread is None:
            return as_mixture(x_hat)
        from src.diffusion.flow_matching.gm_spread import point_mixture
        return point_mixture(x_hat, self.spread(t_cur.float().reshape(1)), 1 - t_cur.float())

    def _substep_sigmas(self, sigma_src, sigma_end):
        """The loop's sigma grid s_j = sigma_src + (sigma_end - sigma_src) * (j / n), j = 0..n, as one (n + 1,) tensor in three device ops (review_gmflow_hn F11).
        Bitwise equal to the loop's values: j / n is rounded from the same double to fp32 on the host, and the multiply and the add are the same fp32 ops
        (tools/gmflow/test_gmflow.py T25)."""
        n, dev = self.substeps, sigma_src.device
        w = self._grid_w.get((n, dev))
        if w is None:
            w = self._grid_w[(n, dev)] = torch.tensor([j / n for j in range(n + 1)], dtype=torch.float32).to(dev)
        return (sigma_src.reshape(1) + (sigma_end - sigma_src).reshape(1) * w)

    def _fused_substeps(self, gm, sigma_src, sigma_end, sig_src_b, x):
        """All sub-steps of one call in one Triton launch (gm_ode_triton.py), on the sigma grid of the loop below."""
        from src.diffusion.flow_matching.gm_ode_triton import gm_ode_substeps_fused
        from src.diffusion.flow_matching.gmflow import logstd_x
        sigmas = self._substep_sigmas(sigma_src, sigma_end)
        s_x = logstd_x(gm, sig_src_b.unsqueeze(1)).exp()
        return gm_ode_substeps_fused(gm["means"], gm["logweights"], s_x, x, sigmas)
