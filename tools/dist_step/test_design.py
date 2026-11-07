#!/usr/bin/env python3
"""Checks for toy_design.py before the runs. usage: test_design.py (runs on CPU, under a minute)."""
import importlib.util
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))


def load(arm, dataset="coupled"):
    sys.argv = ["toy_design.py", dataset, arm, os.devnull, "10", "0"]
    spec = importlib.util.spec_from_file_location("toy_design_" + arm.replace(".", "_"), os.path.join(here, "toy_design.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1. The Gaussian head: the Woodbury mean equals the dense formula, and Matheron's rule has that mean and the dense posterior covariance.
g = load("gauss_r4")
o = torch.randn(1, g.n_out()) * 0.3
x = torch.randn(1, g.D, g.DT)
t = torch.tensor([0.4])
mu, s2, U = g.gauss_parts(o)
S = torch.diag(s2[0]) + U[0] @ U[0].T
a, b = 0.4, 0.6
A = a ** 2 * S + b ** 2 * torch.eye(g.DD)
dense = mu[0] + a * S @ torch.linalg.solve(A, x.flatten() - a * mu[0])
print("1a Woodbury against dense, max error", (g.gauss_mean(o, x, t).flatten() - dense).abs().max().item())
torch.manual_seed(0)
n = 60000
s = g.gauss_sample(o.repeat(n, 1), x.repeat(n, 1, 1), t.repeat(n)).flatten(1)
post = S - a ** 2 * S @ torch.linalg.solve(A, S)
print("1b sample mean error", (s.mean(0) - dense).abs().max().item(), "covariance error", (torch.cov(s.T) - post).abs().max().item(), "largest entry", post.abs().max().item())

# 2. Each arm trains one step, and generates in each mode, at each grid, with finite values.
for dataset in ("coupled", "embedded"):
    for arm in ("point", "impl_late_m2", "cvae_once_l0.1", "cvae_step_l0.1", "part_k8", "gauss_r0", "gauss_r0d", "gauss_r4", "nll_d", "nll_r4", "flow_w48"):
        m = load(arm, dataset)
        net = m.Net()
        loss = m.loss_fn(net, m.sample_data(64))
        loss.backward()
        none = [k for k, p in net.named_parameters() if p.grad is None]
        res = {}
        for mode in m.MODES:
            for calls, kind in ((1, "uniform"), (50, "uniform"), (8, "logit")):
                res[(mode, calls, kind)] = bool(m.generate(net, 16, calls, mode, kind).isfinite().all())
        print("2", dataset, arm, f"loss {loss.item():.3f} cost {m.COST} modes {m.MODES} finite {all(res.values())} params without grad {none}")

# 3. The logit grid is increasing, from 0 to 1.
g = load("point")
ts = g.grid(50, "logit")
print("3 grid", ts[:3].tolist(), ts[-3:].tolist(), "increasing", bool((ts[1:] > ts[:-1]).all()))

# 4. Parsing of the arm names.
for arm, want in (("gauss_r0", (0, False)), ("gauss_r0d", (0, True)), ("gauss_r4", (4, False)), ("nll_d", (0, True)), ("nll_r4", (4, False))):
    m = load(arm)
    print("4", arm, (m.R, m.DIAG), "ok" if (m.R, m.DIAG) == want else "WRONG")
