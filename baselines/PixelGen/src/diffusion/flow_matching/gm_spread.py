"""A spread for a point model (docs/gmflow_hn.md, H): the point model's x prediction x_hat becomes a one-Gaussian mixture N(x_hat, s_x(t)^2 I), with s_x(t) read
from a table of the model's own error, s_x(t) = sqrt(E||x - x_hat(x_t, t)||^2 / D) (tools/gmflow/calibrate_spread.py). GMODESampler(spread_table=...) then runs the
analytic sub-steps of K = 1 on it. With s_x = 0 the mixture is the one as_mixture builds (log s_u = -inf), so the sampler is the exact step.
"""
import json
import os
import re
import sys

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
        return self.interp64(t).float()

    def interp64(self, t):
        """s_x(t) in fp64 (the table's precision)."""
        tt, ss = self.t.to(t.device), self.s.to(t.device)
        q = t.double().clamp(tt[0], tt[-1])
        i = torch.searchsorted(tt, q.reshape(-1), right=True).clamp(1, len(tt) - 1).view(q.shape)
        t0, t1, s0, s1 = tt[i - 1], tt[i], ss[i - 1], ss[i]
        return s0 + (s1 - s0) * (q - t0) / (t1 - t0)

    def check_bound(self, call_ts):
        """The guard of the owner's decision of 2026-11-06 (review_gmflow_hn F2): at every call time t > 0 the sampler reads, s_x(t) <= (1 - t) / t, the error of the
        estimator x_t / t (the table's s_x_trivial). An exact posterior mean is never worse than that estimator, so the posterior std is at most (1 - t) / t; a table
        above it at a call carries the model's error floor, not uncertainty. No clipping and no scaling: the evaluation fails instead."""
        bad = []
        for t in torch.as_tensor(call_ts).reshape(-1).tolist():
            if t <= 0:
                continue
            s_x, bound = float(self.interp64(torch.tensor([t], dtype=torch.float64))[0]), (1 - t) / t   # both in fp64 at the sampler's t
            if s_x > bound:
                bad.append(f"t = {t:.4f}: s_x {s_x:.4f} > (1 - t) / t = {bound:.4f}")
        if bad:
            raise ValueError(f"spread table {self.path} exceeds the bound s_x(t) <= (1 - t) / t at {len(bad)} call(s): " + "; ".join(bad)
                             + ". Use fewer calls (gm_ode_cal runs at 8 and 4 calls; owner 2026-11-06: no clipped or scaled tables).")


def check_table_ckpt(path, step):
    """The table must come from the evaluated checkpoint (review_gmflow_hn F1): basename(table["ckpt"]) names ms-step=<step>."""
    ck = json.load(open(path)).get("ckpt")
    m = re.search(r"ms-step=0*(\d+)", os.path.basename(ck or ""))
    if not m or int(m.group(1)) != int(step):
        raise ValueError(f"spread table {path} was calibrated on {ck}, not on ms-step={int(step):06d}")


def point_mixture(x_hat, s_x, sigma):
    """A point model's output x_hat (B, C, H, W) as the mixture N(x_hat, s_x^2 I): K = 1, log s_u = log s_x - log sigma, so that logstd_x (gmflow.py) returns
    log s_x. s_x: a scalar or (B,) tensor; sigma = 1 - t of the call. s_x = 0 gives log s_u = -inf, the mixture of as_mixture."""
    x = x_hat.float()
    B = x.shape[0]
    s_x = torch.as_tensor(s_x, dtype=torch.float32, device=x.device).reshape(-1).expand(B).reshape(B, 1, 1, 1, 1)
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)[:1].reshape(1, 1, 1, 1, 1)
    logstd = torch.log(s_x) - torch.log(sigma.clamp(min=EPS))
    return dict(means=x[:, None], logweights=torch.zeros(B, 1, 1, *x.shape[2:], device=x.device), logstd=logstd)


def main(argv):
    """jobs/eval_gmflow.sh, before the prediction: check <table> <step> <calls>: the table belongs to ms-step=<step>, and it satisfies the bound at the calls of
    GMODESampler(num_steps=<calls>) with the eval's sampler settings (gmflow_ode_sample.yaml: timeshift 1, last_step null); the sampler itself checks the bound again."""
    assert len(argv) == 4 and argv[0] == "check", "usage: gm_spread.py check <table> <step> <calls>"
    from src.diffusion.base.guidance import simple_guidance_fn
    from src.diffusion.flow_matching.sampling_gmflow import GMODESampler
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    table, step, calls = argv[1], int(argv[2]), int(argv[3])
    try:
        check_table_ckpt(table, step)
        GMODESampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=calls, guidance=1.0, timeshift=1.0,
                     spread_table=table)
    except ValueError as e:
        print(f"refused: {e}")
        return 2
    print(f"table ok: ckpt ms-step={step:06d}, bound holds at the {calls} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
