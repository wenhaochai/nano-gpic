"""A spread for a point model (docs/gmflow_hn.md, H): the point model's x prediction x_hat becomes a one-Gaussian mixture N(x_hat, s_x(t)^2 I), with s_x(t) read
from a table of the model's own error, s_x(t) = sqrt(E||x - x_hat(x_t, t)||^2 / D) (tools/gmflow/calibrate_spread.py). GMODESampler(spread_table=...) then runs the
analytic sub-steps of K = 1 on it. With s_x = 0 the mixture is the one as_mixture builds (log s_u = -inf), so the sampler is the exact step.
"""
import json

import torch

from src.diffusion.flow_matching.gmflow import EPS


class SpreadTable:
    """s_x(t), linear in t between the table's points and constant beyond its ends (as numpy.interp). JSON: {"t": [increasing], "s_x": [...], ...}."""
    def __init__(self, path):
        d = json.load(open(path))
        self.path = path
        self.t = torch.tensor(d["t"], dtype=torch.float64)
        self.s = torch.tensor(d["s_x"], dtype=torch.float64)
        assert self.t.ndim == 1 and self.t.shape == self.s.shape and len(self.t) >= 2, (self.t.shape, self.s.shape)
        assert bool((self.t[1:] > self.t[:-1]).all()), "the table's t must increase"
        assert bool((self.s >= 0).all()) and bool(torch.isfinite(self.s).all()), "s_x must be finite and >= 0"

    def __call__(self, t):
        """t: a tensor of any shape; returns s_x(t) in fp32 on t's device."""
        tt, ss = self.t.to(t.device), self.s.to(t.device)
        q = t.double().clamp(tt[0], tt[-1])
        i = torch.searchsorted(tt, q.reshape(-1), right=True).clamp(1, len(tt) - 1).view(q.shape)
        t0, t1, s0, s1 = tt[i - 1], tt[i], ss[i - 1], ss[i]
        return (s0 + (s1 - s0) * (q - t0) / (t1 - t0)).float()


def point_mixture(x_hat, s_x, sigma):
    """A point model's output x_hat (B, C, H, W) as the mixture N(x_hat, s_x^2 I): K = 1, log s_u = log s_x - log sigma, so that logstd_x (gmflow.py) returns
    log s_x. s_x: a scalar or (B,) tensor; sigma = 1 - t of the call. s_x = 0 gives log s_u = -inf, the mixture of as_mixture."""
    x = x_hat.float()
    B = x.shape[0]
    s_x = torch.as_tensor(s_x, dtype=torch.float32, device=x.device).reshape(-1).expand(B).reshape(B, 1, 1, 1, 1)
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)[:1].reshape(1, 1, 1, 1, 1)
    logstd = torch.log(s_x) - torch.log(sigma.clamp(min=EPS))
    return dict(means=x[:, None], logweights=torch.zeros(B, 1, 1, *x.shape[2:], device=x.device), logstd=logstd)
