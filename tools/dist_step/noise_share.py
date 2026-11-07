#!/usr/bin/env python3
"""How much of the final sample's variance comes from the fresh draw at each step, with an exact posterior (no model error)?
Prior: a mixture of Gaussians in 16 dimensions, so p(x | x_t) is closed form. Step: x + dt / (1 - t) (x_hat - x), x_hat ~ p(x | x_t).
This step has the right marginal at any step size. From one x_0, repeat the path with new draws: the variance across repeats, over the total variance,
is the share of the fresh draws. The rest is set by x_0 alone. Also the mean path: x_hat = E[x | x_t], the ODE with Euler.
usage: noise_share.py
"""
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
DIM, K, S2 = 16, 8, 0.01
MU = torch.randn(K, DIM)
MU = MU / MU.norm(dim=1, keepdim=True) * DIM ** 0.5


def posterior(x, t):
    """Responsibilities, component means and the common variance of p(x0 | x_t), x_t = t x0 + (1 - t) eps."""
    a, b = t, 1 - t
    v = a ** 2 * S2 + b ** 2
    r = x[:, None] - a * MU                                   # (n, K, DIM)
    logw = -(r ** 2).sum(-1) / (2 * v)
    w = logw.softmax(-1)
    m = MU + (a * S2 / v) * r
    c = S2 * b ** 2 / v
    return w, m, c


def run(x0, steps, fresh):
    x = x0.clone()
    n = len(x)
    ts = torch.linspace(0, 1, steps + 1)
    for i in range(steps):
        t, dt = ts[i].item(), (ts[i + 1] - ts[i]).item()
        w, m, c = posterior(x, t)
        if fresh:
            k = torch.multinomial(w, 1)[:, 0]
            xh = m[torch.arange(n), k] + c ** 0.5 * torch.randn(n, DIM)
        else:
            xh = (w[..., None] * m).sum(1)
        x = x + dt / (1 - t) * (xh - x)
    return x


def energy(a, b):
    n, m = len(a), len(b)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).sum() / (n * (n - 1)) - torch.cdist(b, b).sum() / (m * (m - 1))).item()


real = MU[torch.randint(K, (4000,))] + S2 ** 0.5 * torch.randn(4000, DIM)
print(f"{'steps':>6s} {'share of the fresh draws':>25s} {'same mode from one x_0':>23s} {'ED fresh':>9s} {'ED mean path':>13s}")
for steps in (1, 2, 4, 8, 16, 50, 200, 800):
    x0s = torch.randn(400, DIM)
    rep = 200
    xs = run(x0s.repeat_interleave(rep, 0), steps, True).view(400, rep, DIM)
    within = xs.var(1).sum(-1).mean().item()
    total = xs.flatten(0, 1).var(0).sum().item()
    near = torch.cdist(xs.flatten(0, 1), MU).argmin(1).view(400, rep)
    same = (torch.stack([torch.bincount(near[i], minlength=K).max() for i in range(400)]).float() / rep).mean().item()
    print(f"{steps:6d} {within / total:25.4f} {same:23.3f} {energy(run(torch.randn(4000, DIM), steps, True), real):9.4f} {energy(run(torch.randn(4000, DIM), steps, False), real):13.4f}", flush=True)
