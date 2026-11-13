#!/usr/bin/env python3
"""Toy test of a model whose every step outputs a distribution instead of a velocity. CPU only.

The backbone is the same in all arms. Only the last layer and the loss differ:
  point      x-prediction, v-space L2: the JiT recipe.
  mix_t      each token gets K anchors, logits, variances: a mixture prior; the output is its posterior mean given (x_t, t) by Bayes'
             rule; v-space L2 at level t only.
  mix_multi  the same head; v-space L2 at level t and at m lower noise levels from one backbone pass (x_s from the bridge x_t -> x).
  mix_nll    K anchors, logits, variances are a mixture posterior q(x | x_t), trained by negative log-likelihood (the GMFlow form).
  wiener     mix_t with K = 1: one anchor and one variance per token, so only a learned skip from x_t to the output.
  mix_global mix_t with one mode choice per sample: the logits and likelihoods of all tokens add up before the softmax.
  mix_global_multi  mix_global with the loss of mix_multi: a joint output trained at many noise levels per backbone pass.
  tangent    each token gets one anchor and R directions: a Gaussian prior with covariance U U^T and no isotropic part, so the skip from
             x_t passes only inside the span of U and noise off that span cannot leak into the output.
  tangent_mix  the same with K anchors that share U.
Sampling: N backbone calls; between calls M Euler substeps that use only the head output of the last call (M = 1: plain Euler).

Data: an 8-cell checkerboard per token, D tokens.
  single       D = 1
  independent  D = 8, each token picks its cell alone: 8^8 modes that factor over tokens
  coupled      D = 8, all tokens pick one cell: 8 modes that a per-token mixture cannot state
  manifold     D = 8, one latent angle sets all tokens: a 1-D curve in 16 dimensions, closed form, no discrete modes
  embedded     the same curve with 32-dimensional tokens: a low-dimensional manifold inside each token, as patches of pixels are

usage: toy_head.py <dataset> <arm> <out.json> [train steps] [seed]
"""
import copy
import json
import math
import sys
import time

import torch
import torch.nn as nn

torch.set_num_threads(1)
dataset, arm, out = sys.argv[1:4]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
torch.manual_seed(SEED)
D = 1 if dataset == "single" else 8
DT = 32 if dataset == "embedded" else 2                     # dimension of one token
K = 1 if arm in ("point", "wiener", "tangent") else 8
R = 4 if arm.startswith("tangent") else 0                   # tangent directions per token, shared by the token's anchors
M_INNER = 4
T_EPS = 0.05
CELLS = torch.tensor([(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0], dtype=torch.float32)


HARM = torch.arange(1, 9, dtype=torch.float32)            # manifold: token i turns i + 1 times per turn of the one latent angle
PHASE = torch.tensor([0.3, 2.1, 4.4, 1.2, 5.6, 3.3, 0.9, 2.7])


# embedded: the same curve, each token's circle in a random 2-D plane of a 32-D token, with the same variance per dimension as the 2-D
# token up to a constant: a low-dimensional manifold in a high-dimensional token, as patches of pixels are
QS = torch.linalg.qr(torch.randn(D, 32, 2, generator=torch.Generator().manual_seed(123))).Q if dataset == "embedded" else None


def curve(th):
    ang = th * HARM + PHASE
    xy = 0.75 * torch.stack([ang.cos(), ang.sin()], -1)
    return 4 * (QS @ xy[..., None])[..., 0] if dataset == "embedded" else xy


def sample_data(n):
    if dataset in ("manifold", "embedded"):
        return curve(torch.rand(n, 1) * 2 * math.pi) + 0.01 * torch.randn(n, D, DT)
    idx = torch.randint(8, (n, 1)).expand(n, D) if dataset == "coupled" else torch.randint(8, (n, D))
    return (CELLS[idx] + torch.rand(n, D, 2)) / 2 - 1          # (n, D, 2) in [-1, 1]


class Net(nn.Module):
    def __init__(self, width=256, depth=4):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(6) * math.pi)
        self.inp = nn.Linear(D * DT + 12, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Linear(width, D * K * (DT + 2) + D * R * DT)

    def forward(self, x, t):
        f = t[:, None] * self.freq
        h = self.inp(torch.cat([x.flatten(1), f.sin(), f.cos()], 1))
        for b in self.blocks:
            h = h + b(h)
        o = self.out(h)
        if R:
            mix, u = o[:, :D * K * (DT + 2)].view(-1, D, K, DT + 2), o[:, D * K * (DT + 2):].view(-1, D, R, DT)
            return mix[..., :DT], mix[..., DT], u              # anchors, logits, directions
        o = o.view(-1, D, K, DT + 2)
        return o[..., :DT], o[..., DT], o[..., DT + 1]         # anchors, logits, log variances


def tangent_mean(mu, logit, U, x, t):
    """E[x0 | x] for a per-token prior sum_k pi_k N(mu_k, U U^T), with x = t x0 + (1 - t) eps, by the Woodbury identity."""
    a, b2 = t.view(-1, 1, 1, 1), ((1 - t) ** 2).clamp_min(1e-4).view(-1, 1, 1, 1)
    A = a ** 2 * (U @ U.transpose(-1, -2)) + b2 * torch.eye(R)  # (n, D, R, R)
    L = torch.linalg.cholesky(A)
    r = x[:, :, None, :] - a * mu                               # (n, D, K, DT)
    Ur = r @ U.transpose(-1, -2)                                # (n, D, K, R)
    sol = torch.cholesky_solve(Ur.transpose(-1, -2), L).transpose(-1, -2)
    m = mu + a * (sol @ U)
    if K == 1:
        return m[:, :, 0]
    quad = ((r ** 2).sum(-1) - a[..., 0] ** 2 * (Ur * sol).sum(-1)) / b2[..., 0]     # the log determinant is shared by the anchors
    return ((logit - 0.5 * quad).softmax(-1)[..., None] * m).sum(2)


def bayes_mean(mu, logit, log_s2, x, t):
    """E[x0 | x] for a per-token prior sum_k pi_k N(mu_k, s_k^2 I), with x = t x0 + (1 - t) eps."""
    if R:
        return tangent_mean(mu, logit, log_s2, x, t)
    a, b = t.view(-1, 1, 1), (1 - t).view(-1, 1, 1)
    s2 = log_s2.exp() + 1e-6
    v = a ** 2 * s2 + b ** 2
    r = x[:, :, None, :] - a[..., None] * mu
    logw = logit - (r ** 2).sum(-1) / (2 * v) - DT / 2 * v.log()
    w = logw.sum(1, keepdim=True).softmax(-1) if arm.startswith("mix_global") else logw.softmax(-1)
    m = mu + (a * s2 / v)[..., None] * r
    return (w[..., None] * m).sum(2)


def tilted_mean(mu, logit, log_c, x0, t0, x, t):
    """Mean at (x, t) of q(x0 | x_t0) N(x; t x0, (1-t)^2) / N(x_t0; t0 x0, (1-t0)^2), for a mixture q, from q alone."""
    c = log_c.exp() + 1e-6
    k = lambda u: (u / (1 - u)) ** 2
    lam = 1 / c + k(t).view(-1, 1, 1) - k(t0).view(-1, 1, 1)
    eta = mu / c[..., None] + (t / (1 - t) ** 2).view(-1, 1, 1, 1) * x[:, :, None, :] - (t0 / (1 - t0) ** 2).view(-1, 1, 1, 1) * x0[:, :, None, :]
    cn = 1 / lam
    mn = cn[..., None] * eta
    logw = logit.log_softmax(-1) + DT / 2 * (cn / c).log() + 0.5 * ((mn ** 2).sum(-1) / cn - (mu ** 2).sum(-1) / c)
    return (logw.softmax(-1)[..., None] * mn).sum(2)


def vweight(t):
    return 1 / (1 - t).clamp_min(T_EPS).view(-1, 1, 1) ** 2


def loss_fn(net, x):
    n = x.size(0)
    t = (torch.randn(n) * 0.8 - 0.8).sigmoid()
    eps = torch.randn_like(x)
    x_t = t.view(-1, 1, 1) * x + (1 - t).view(-1, 1, 1) * eps
    mu, logit, log_s2 = net(x_t, t)
    if arm == "point":
        return (vweight(t) * (mu[:, :, 0] - x) ** 2).mean()
    if arm == "mix_nll":
        c = log_s2.exp() + 1e-6
        return -(logit.log_softmax(-1) - ((x[:, :, None, :] - mu) ** 2).sum(-1) / (2 * c) - DT / 2 * c.log()).logsumexp(-1).mean()
    loss = (vweight(t) * (bayes_mean(mu, logit, log_s2, x_t, t) - x) ** 2).mean()
    if arm in ("mix_multi", "mix_global_multi"):
        st = ((1 - t) / t).view(-1, 1, 1)
        inner = 0
        for _ in range(M_INNER):
            tp = t + (1 - t) * torch.rand(n) * 0.98            # lower noise than t
            sp = ((1 - tp) / tp).view(-1, 1, 1)
            r = (sp / st) ** 2
            y = x + sp ** 2 / st * eps + sp * (1 - r).clamp_min(0).sqrt() * torch.randn_like(x)      # bridge x_t -> x, in x + sigma eps form
            inner = inner + (vweight(tp) * (bayes_mean(mu, logit, log_s2, tp.view(-1, 1, 1) * y, tp) - x) ** 2).mean()
        loss = loss + inner / M_INNER
    return loss


@torch.no_grad()
def generate(net, n, calls, sub):
    x = torch.randn(n, D, DT)
    ts = torch.linspace(0, 1, calls + 1)
    for i in range(calls):
        t0 = ts[i].expand(n)
        mu, logit, log_s2 = net(x, t0)
        x0 = x
        dt = (ts[i + 1] - ts[i]) / sub
        for j in range(sub):
            t = t0 + j * dt
            if arm == "point":
                pred = mu[:, :, 0]
            elif arm == "mix_nll":
                pred = (logit.softmax(-1)[..., None] * mu).sum(2) if j == 0 else tilted_mean(mu, logit, log_s2, x0, t0, x, t)
            else:
                pred = bayes_mean(mu, logit, log_s2, x, t)
            x = x + dt * (pred - x) / (1 - t).clamp_min(T_EPS).view(-1, 1, 1)
    return x


def energy_distance(a, b):
    a, b = a.flatten(1), b.flatten(1)
    n, m = len(a), len(b)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).sum() / (n * (n - 1)) - torch.cdist(b, b).sum() / (m * (m - 1))).item()


def validity(x):
    if dataset in ("manifold", "embedded"):                    # distance to the curve and coverage along the curve
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
    """The posterior mean's squared error by noise level: the point every arm must get right for the ODE."""
    out_ = {}
    x = sample_data(n)
    for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        t = torch.full((n,), t_val)
        x_t = t_val * x + (1 - t_val) * torch.randn_like(x)
        mu, logit, log_s2 = net(x_t, t)
        if arm == "point":
            pred = mu[:, :, 0]
        elif arm == "mix_nll":
            pred = (logit.softmax(-1)[..., None] * mu).sum(2)
        else:
            pred = bayes_mean(mu, logit, log_s2, x_t, t)
        out_[t_val] = ((pred - x) ** 2).mean().item()
    return out_


net = Net()
ema = copy.deepcopy(net)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
rows, t_start = [], time.time()
evals = {s for s in (500, 1000, 2000, 5000, 10000, 20000, 50000) if s <= STEPS} | {STEPS}
for step in range(1, STEPS + 1):
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
        row = {"step": step, "dataset": dataset, "arm": arm, "seed": SEED, "seconds": time.time() - t_start, "val": val_loss(ema), "gen": []}
        for calls in (1, 2, 4, 8, 50):
            for sub in ((1,) if arm == "point" else (1, 16)):
                g = generate(ema, 2000, calls, sub)
                row["gen"].append({"calls": calls, "sub": sub, "ed": energy_distance(g, real), **validity(g)})
        rows.append(row)
        with open(out, "w") as f:
            json.dump(rows, f)
        print(dataset, arm, step, f"{row['seconds']:.0f} s", " ".join(f"{g['calls']}/{g['sub']}:{g['ed']:.4f}" for g in row["gen"]), flush=True)
