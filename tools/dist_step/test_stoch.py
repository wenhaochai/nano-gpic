#!/usr/bin/env python3
"""CPU checks of the stochastic step in EulerSamplerJiT. The data is a Gaussian mixture in one dimension per sample, and the network is its exact posterior mean, so the sampler must return the data's law for any eta, on a fine grid.
Run on a vis node from baselines/PixelGen with PYTHONPATH=. : python ../../tools/dist_step/test_stoch.py"""
import torch

from src.diffusion.flow_matching.sampling import EulerSamplerJiT
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.base.guidance import simple_guidance_fn  # the guidance function of the configs

W = torch.tensor([0.55, 0.45])
M = torch.tensor([-0.55, 0.6])
S = torch.tensor([0.10, 0.12])


def posterior_mean(x, t, y=None):
    """E[x | x_t] for x ~ mixture, x_t = t x + (1 - t) eps."""
    t = t.view(-1, 1, 1, 1)
    s2 = S.view(1, 1, 1, 1, 2) ** 2
    n2 = (1 - t).clamp_min(1e-6)[..., None] ** 2
    v = t[..., None] ** 2 * s2 + n2
    d = x[..., None] - t[..., None] * M.view(1, 1, 1, 1, 2)
    log_r = W.log() - 0.5 * (d ** 2 / v + v.log())
    r = (log_r - log_r.max(-1, keepdim=True)[0]).exp()
    r = r / r.sum(-1, keepdim=True)
    mean = M.view(1, 1, 1, 1, 2) + t[..., None] * s2 / v * d
    return (r * mean).sum(-1)


def run(steps, seed=0, **kw):
    torch.manual_seed(seed)
    sampler = EulerSamplerJiT(scheduler=LinearScheduler(), num_steps=steps, guidance=1.0, guidance_fn=simple_guidance_fn, **kw)
    noise = torch.randn(200000, 1, 1, 1)
    xs, _ = sampler._impl_sampling(posterior_mean, noise, None, None)
    return xs[-1].flatten()


def summary(x):
    left = (x < 0).float().mean().item()
    return left, x.mean().item(), x.std().item()


true = torch.cat([M[0] + S[0] * torch.randn(110000), M[1] + S[1] * torch.randn(90000)])
print("data       share of the left mode %.4f, mean %.4f, std %.4f" % summary(true))
base = run(50)
assert (base == run(50)).all()
for eta in (0.0, 0.5, 1.0):
    for steps in (50, 400):
        x = run(steps, eta=eta) if eta else run(steps)
        print(f"eta {eta} steps {steps:3d}: share of the left mode %.4f, mean %.4f, std %.4f" % summary(x))
        if steps == 400:
            l, m, sd = summary(x)
            assert abs(l - 0.55) < 0.01 and abs(m - summary(true)[1]) < 0.02 and abs(sd - true.std().item()) < 0.02, (eta, l, m, sd)
# the path with eta -> 0 is the official path, up to the clamp near t = 1
a, b = run(50), run(50, land=True)
print("land: largest change of a sample %.4f, std of the change %.4f" % ((a - b).abs().max().item(), (a - b).std().item()))
assert (a - b).abs().max() < 0.2
print("all checks passed")
