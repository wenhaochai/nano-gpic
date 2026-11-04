#!/usr/bin/env python3
"""Does the head's flow give a mode its true probability? In 768 dimensions a mode is decided at a very small head time (the signal adds over 768 values), where the official logit-normal(-0.8, 0.8) rarely samples. Each row: the head alone, trained on 8 modes with known shares; total variation between the drawn shares and the true shares, and rms of a draw to its mode.
Run on a vis node from baselines/PixelGen with PYTHONPATH=. : python ../../tools/ar/head_modes.py"""
import sys
import torch

from src.models.transformer.AR_JiT import TokenFlowHead

D = 768
torch.manual_seed(0)
modes = torch.randn(8, D).clamp(-2, 2) * 0.5
shares = torch.tensor([.4, .2, .1, .1, .08, .06, .04, .02])
cond = torch.zeros(1, 64)
WIDTH, DEPTH, STEPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


def draw_tau(kind, n):
    if kind == "official":
        return (torch.randn(n) * 0.8 - 0.8).sigmoid()
    if kind == "uniform":
        return torch.rand(n)
    if kind == "wide":
        return (torch.randn(n) * 1.6 - 0.8).sigmoid()
    raise ValueError(kind)


def grid_of(kind, steps):
    if kind == "uniform":
        return torch.linspace(0, 1, steps + 1)
    if kind == "dense_start":   # uniform in tau^2: more steps at small tau
        return torch.linspace(0, 1, steps + 1) ** 2
    raise ValueError(kind)


def sample(head, n, grid):
    q = torch.randn(n, D)
    c = cond.repeat(n, 1)
    with torch.no_grad():
        for j in range(len(grid) - 1):
            hat = head(q, grid[j].repeat(n), c)
            q = hat if j == len(grid) - 2 else q + (grid[j + 1] - grid[j]) * (hat - q) / (1 - grid[j]).clamp_min(0.05)
    return q


for kind in ("official", "wide", "uniform"):
    torch.manual_seed(1)
    head = TokenFlowHead(D, 64, WIDTH, DEPTH)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for step in range(STEPS):
        idx = torch.multinomial(shares, 256, replacement=True)
        target = modes[idx]
        tau = draw_tau(kind, 256)
        eta = torch.randn(256, D)
        q = tau[:, None] * target + (1 - tau[:, None]) * eta
        hat = head(q, tau, cond.repeat(256, 1))
        loss = (((hat - target) / (1 - tau[:, None]).clamp_min(0.05)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for gk, gs in (("uniform", 32), ("dense_start", 32), ("uniform", 128)):
        x = sample(head, 4000, grid_of(gk, gs))
        d = torch.cdist(x, modes) / D ** 0.5
        nn, who = d.min(1)
        got = torch.bincount(who, minlength=8).float() / len(who)
        print(f"train tau {kind:8s} grid {gk:11s} {gs:3d} steps: TV {0.5 * (got - shares).abs().sum().item():.3f}, rms to the mode {nn.mean().item():.3f}, shares {[round(v, 2) for v in got.tolist()]}", flush=True)
