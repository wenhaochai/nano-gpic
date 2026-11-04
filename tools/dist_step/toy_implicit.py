#!/usr/bin/env python3
"""Toy test of an implicit sampler: x_hat = g(x_t, t, xi), a sample of p(x | x_t) instead of its mean. CPU only.

Why no discretization error in principle: if x_hat ~ p(x | x_t), then x_t + dt / (1 - t) (x_hat - x_t) = (t + dt) x_hat + (1 - t - dt) eps_hat, with
eps_hat = (x_t - t x_hat) / (1 - t), and (x_hat, eps_hat) has the law of (x, eps) given x_t. So the next state has the right marginal at any step size.

Arms, name = impl_<site>_m<m>[_<loss>]:
  site   input: xi joins the input, so each sample costs a whole forward. late: the first blocks run once per x_t, xi joins before the last blocks.
  m      samples of g per x_t; the training compute is charged, and each arm trains for the same compute as the point arm's 10000 steps.
  loss   es: generalized energy score over the whole sample, beta = 1, lambda = 1 (strictly proper), weight 1 / (1 - t).
         tok: the same score token by token (proper for each token's marginal, blind to the joint).
         l2: es plus the v-space L2 of the mean of the m samples (the point arm's loss, the same weight as its use there).
         b15, b18: es with beta = 1.5 or 1.8 on w (g - x) (strictly proper for beta < 2; the gradient shrinks near the target, unlike beta = 1).
  point  the JiT recipe, as the reference: x-prediction, v-space L2, Euler with the mean.

usage: toy_implicit.py <dataset> <arm> <out.json> [compute in point-arm steps] [seed]
"""
import json
import math
import sys
import time
import copy

import torch
import torch.nn as nn

torch.set_num_threads(1)
dataset, arm, out = sys.argv[1:4]
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
torch.manual_seed(SEED)
D = 1 if dataset == "single" else 8
DT = 32 if dataset == "embedded" else 2
T_EPS = 0.05
XI = 32
if arm == "point":
    SITE, M, LOSS = "none", 1, "point"
else:
    parts = arm.split("_")
    SITE, M, LOSS = parts[1], int(parts[2][1:]), (parts[3] if len(parts) > 3 else "es")
WIDTH, DEPTH, SPLIT = 256, 4, 2                            # late: SPLIT blocks once, the others per sample
COST = {"none": 1.0, "input": float(M), "late": (SPLIT + M * (DEPTH - SPLIT)) / DEPTH}[SITE]     # blocks run per x_t, against the point arm

CELLS = torch.tensor([(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0], dtype=torch.float32)
HARM = torch.arange(1, 9, dtype=torch.float32)
PHASE = torch.tensor([0.3, 2.1, 4.4, 1.2, 5.6, 3.3, 0.9, 2.7])
QS = torch.linalg.qr(torch.randn(D, 32, 2, generator=torch.Generator().manual_seed(123))).Q if dataset == "embedded" else None


def curve(th):
    ang = th * HARM + PHASE
    xy = 0.75 * torch.stack([ang.cos(), ang.sin()], -1)
    return 4 * (QS @ xy[..., None])[..., 0] if dataset == "embedded" else xy


def sample_data(n):
    if dataset in ("manifold", "embedded"):
        return curve(torch.rand(n, 1) * 2 * math.pi) + 0.01 * torch.randn(n, D, DT)
    idx = torch.randint(8, (n, 1)).expand(n, D) if dataset == "coupled" else torch.randint(8, (n, D))
    return (CELLS[idx] + torch.rand(n, D, 2)) / 2 - 1


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(6) * math.pi)
        self.inp = nn.Linear(D * DT + 12 + (XI if SITE == "input" else 0), WIDTH)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH)) for _ in range(DEPTH))
        self.xi = nn.Linear(XI, WIDTH) if SITE == "late" else None
        self.out = nn.Linear(WIDTH, D * DT)

    def forward(self, x, t, m=1):
        """(n, m, D, DT): m samples for each of the n inputs."""
        n = len(x)
        f = t[:, None] * self.freq
        base = torch.cat([x.flatten(1), f.sin(), f.cos()], 1)
        if SITE == "input":
            h = self.inp(torch.cat([base.repeat_interleave(m, 0), torch.randn(n * m, XI)], 1))
            for b in self.blocks:
                h = h + b(h)
        else:
            h = self.inp(base)
            for b in self.blocks[:SPLIT]:
                h = h + b(h)
            h = h.repeat_interleave(m, 0)
            if SITE == "late":
                h = h + self.xi(torch.randn(n * m, XI))
            for b in self.blocks[SPLIT:]:
                h = h + b(h)
        return self.out(h).view(n, m, D, DT)


def loss_fn(net, x):
    n = len(x)
    t = (torch.randn(n) * 0.8 - 0.8).sigmoid()
    x_t = t.view(-1, 1, 1) * x + (1 - t).view(-1, 1, 1) * torch.randn_like(x)
    g = net(x_t, t, M)                                          # (n, m, D, DT)
    w = 1 / (1 - t).clamp_min(T_EPS)
    if LOSS == "point":
        return ((w.view(-1, 1, 1) * (g[:, 0] - x)) ** 2).mean()
    if LOSS == "tok":                                           # per token: mean over tokens of the score of that token
        a = (g - x[:, None]).norm(dim=-1).mean(1)               # (n, D)
        d = (g[:, :, None] - g[:, None]).norm(dim=-1)           # (n, m, m, D)
        b = d.sum((1, 2)) / (M * (M - 1))
        return (w[:, None] * (a - 0.5 * b)).mean()
    gf, xf = g.flatten(2), x.flatten(1)[:, None]
    if LOSS in ("b15", "b18"):
        beta = {"b15": 1.5, "b18": 1.8}[LOSS]
        a = ((w[:, None, None] * (gf - xf)) ** 2).sum(-1).add(1e-8).pow(beta / 2).mean(1)
        d2 = ((w[:, None, None, None] * (gf[:, :, None] - gf[:, None])) ** 2).sum(-1)
        off = ~torch.eye(M, dtype=torch.bool)
        b = d2[:, off].add(1e-8).pow(beta / 2).mean(1)
        return (a - 0.5 * b).mean()
    a = (gf - xf).norm(dim=-1).mean(1)
    b = torch.cdist(gf, gf).sum((1, 2)) / (M * (M - 1))
    loss = (w * (a - 0.5 * b)).mean()
    if LOSS == "l2":
        loss = loss + ((w.view(-1, 1, 1) * (g.mean(1) - x)) ** 2).mean()
    return loss


@torch.no_grad()
def generate(net, n, calls):
    x = torch.randn(n, D, DT)
    ts = torch.linspace(0, 1, calls + 1)
    for i in range(calls):
        t, dt = ts[i].expand(n), ts[i + 1] - ts[i]
        pred = net(x, t, 1)[:, 0]
        if arm == "point":                                      # as the official sampler: t_eps clamp
            x = x + dt * (pred - x) / (1 - t).clamp_min(T_EPS).view(-1, 1, 1)
        else:                                                   # an exact step for a posterior sample; the last call returns pred
            x = x + dt * (pred - x) / (1 - t).view(-1, 1, 1)
    return x


def energy_distance(a, b):
    a, b = a.flatten(1), b.flatten(1)
    n, m = len(a), len(b)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).sum() / (n * (n - 1)) - torch.cdist(b, b).sum() / (m * (m - 1))).item()


def validity(x):
    if dataset in ("manifold", "embedded"):
        ths = torch.linspace(0, 2 * math.pi, 4000)
        grid = curve(ths[:, None]).flatten(1)
        d = torch.cdist(x.flatten(1), grid)
        real = torch.cdist((curve(torch.rand(2000, 1) * 2 * math.pi) + 0.01 * torch.randn(2000, D, DT)).flatten(1), grid).min(1).values
        hist = torch.histc(d.argmin(1).float(), bins=64, min=0, max=3999) / len(x) * 64
        return {"dist": d.min(1).values.mean().item(), "dist_real": real.mean().item(), "coverage_min": hist.min().item(), "coverage_max": hist.max().item()}
    g = ((x + 1) * 2).floor().long()
    inside = ((g >= 0) & (g <= 3)).all(-1) & ((g[..., 0] + g[..., 1]) % 2 == 0)
    same = (g == g[:, :1]).all(-1).all(-1)
    c = (CELLS + 0.5) / 2 - 1
    near = torch.cdist(x.flatten(0, 1), c).argmin(1).view(len(x), D)
    within = (x - c[near]).pow(2).mean().sqrt() / (0.5 / math.sqrt(12))
    share = torch.bincount(near.flatten(), minlength=8).float() / near.numel() * 8
    coherent = (near == near[:, :1]).all(1).float().mean()
    return {"inside": inside.float().mean().item(), "same_cell": (inside.all(-1) & same).float().mean().item(), "coherent": coherent.item(),
            "spread": within.item(), "share_min": share.min().item(), "share_max": share.max().item()}


@torch.no_grad()
def val_loss(net, n=5000):
    """Squared error of the mean of 8 samples: how well the posterior mean is learned, with no credit for spread."""
    res = {}
    x = sample_data(n)
    for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        t = torch.full((n,), t_val)
        x_t = t_val * x + (1 - t_val) * torch.randn_like(x)
        pred = net(x_t, t, 8 if arm != "point" else 1).mean(1)
        res[t_val] = ((pred - x) ** 2).mean().item()
    return res


net = Net()
ema = copy.deepcopy(net)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
rows, t_start = [], time.time()
evals = {int(s / COST): s for s in (500, 1000, 2000, 5000, 10000, 20000, 50000) if s <= BUDGET}
evals[int(BUDGET / COST)] = BUDGET
for step in range(1, int(BUDGET / COST) + 1):
    loss = loss_fn(net, sample_data(512))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for p, q in zip(ema.parameters(), net.parameters()):
            p.lerp_(q, 1e-3)
    if step in evals:
        real = sample_data(2000)
        row = {"step": evals[step], "real_steps": step, "dataset": dataset, "arm": arm, "seed": SEED, "seconds": time.time() - t_start, "val": val_loss(ema), "gen": []}
        for calls in (1, 2, 4, 8, 50):
            g = generate(ema, 2000, calls)
            row["gen"].append({"calls": calls, "sub": 1, "ed": energy_distance(g, real), **validity(g)})
        rows.append(row)
        with open(out, "w") as f:
            json.dump(rows, f)
        print(dataset, arm, evals[step], step, f"{row['seconds']:.0f} s", " ".join(f"{g['calls']}:{g['ed']:.4f}" for g in row["gen"]), flush=True)
