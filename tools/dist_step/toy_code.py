#!/usr/bin/env python3
"""Toy test of plan A in depth: a code z per sample, fixed along the sampling path, and how to draw z. CPU only. Data, backbone and recipe as in toy_design.py.

x_hat = g(x_t, t, z). Training: z comes from the encoder, v-space L2, a rate term. Tiny flows on z are trained alongside with stop-gradient, so they do not change the main network.

Arms: code_<encoder>_<site>_<rate>
  encoder  x     q(z | x)                              Exact if the model is exact: a flow conditioned on z, with z ~ q(z).
           xt    q(z | x, x_t, t)                      The form of Variational Rectified Flow Matching.
           pca<k> z = the top k principal components of x, whitened, no noise, no rate term: a fixed linear code, the analogue of a thumbnail.
           chan  a Gaussian channel of fixed capacity: u from the encoder, scaled to the sphere of radius sqrt(d), z = a u + sqrt(1 - a^2) eps, a^2 = 1 - exp(-2 R / d). No KL and no weight:
                 I(x; z) <= R nats by Shannon's bound, whatever the encoder does. The rate spec r<R> sets R. Only the flow can sample z well.
  site     late  z joins before the last 2 blocks (a head on the features before it can see x_t).
           in    z joins at the input, so all blocks see it.
  rate     l<L>  L2 + L * KL / (output dims).
           r<R>  the same, with L adjusted during training to hold KL at R nats.
           none  (pca only).
Sampling modes:
  gauss      z ~ N(0, I), once per path.
  prior      z from an unconditional tiny flow trained on the encoder's z, once per path. This tests whether the failure of gauss is a prior hole.
  head_once  z from a tiny flow conditioned on the features before z (late only) and t, drawn at t = 0, once per path.
  head_step  the same flow, a new z at each step from the current x_t, with the exact step. This is a whole distribution per step: a distribution over the code, then a mean for each code.
  gauss_step z ~ N(0, I) at each step, with the exact step (xt only).
COST charges the encoder at 4%. The flows on z are not charged: in the real model such a flow is under 0.1% of a forward, and here the backbone is too small for the ratio to carry over.

usage: toy_code.py <dataset> <arm> <out.json> [compute in point-arm steps] [seed]
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
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
torch.manual_seed(SEED)
D, DT = 8, (32 if dataset == "embedded" else 2)
DD = D * DT
T_EPS = 0.05
WIDTH, DEPTH, SPLIT = 256, 4, 2
NORM = -0.8, 0.8
FW, FSTEPS = 128, 32

_, ENC, SITE, RATE = arm.split("_")
PCA_K = int(ENC[3:]) if ENC.startswith("pca") else 0
XI = PCA_K or 32
LAM0 = float(RATE[1:]) if RATE.startswith("l") else 1.0
TARGET = float(RATE[1:]) if RATE.startswith("r") and ENC != "chan" else None
CHAN_A = (1 - math.exp(-2 * float(RATE[1:]) / XI)) ** 0.5 if ENC == "chan" else 0.0
COST = 1.0 if PCA_K else 1.04
MODES = {"x": ["gauss", "prior"] + (["head_once", "head_step"] if SITE == "late" else []), "xt": ["gauss", "gauss_step", "prior"], "chan": ["gauss", "prior"]}.get(ENC, ["prior"])

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
    idx = torch.randint(8, (n, 1)).expand(n, D)
    return (CELLS[idx] + torch.rand(n, D, 2)) / 2 - 1


def block(w):
    return nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))


class Flow(nn.Module):
    """A tiny flow on z, conditioned on c (or nothing)."""
    def __init__(self, c_dim):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(6) * math.pi)
        self.c_dim = c_dim
        self.inp = nn.Linear(XI + 12 + c_dim, FW)
        self.blocks = nn.ModuleList(block(FW) for _ in range(3))
        self.out = nn.Linear(FW, XI)

    def forward(self, z, tau, c=None):
        f = tau[:, None] * self.freq
        parts = [z, f.sin(), f.cos()] + ([c] if self.c_dim else [])
        h = self.inp(torch.cat(parts, 1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)

    def loss(self, z, c=None):
        tau = torch.rand(len(z))
        eta = torch.randn_like(z)
        y = tau[:, None] * z + (1 - tau[:, None]) * eta
        return ((self(y, tau, c) - (z - eta)) ** 2).mean()

    @torch.no_grad()
    def sample(self, n, c=None):
        z = torch.randn(n, XI)
        for j in range(FSTEPS):
            z = z + self(z, torch.full((n,), j / FSTEPS), c) / FSTEPS
        return z


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(6) * math.pi)
        self.inp = nn.Linear(DD + 12 + (XI if SITE == "in" else 0), WIDTH)
        self.blocks = nn.ModuleList(block(WIDTH) for _ in range(DEPTH))
        self.out = nn.Linear(WIDTH, DD)
        if SITE == "late":
            self.xi = nn.Linear(XI, WIDTH)
        if not PCA_K:
            self.enc = nn.Sequential(nn.Linear(DD if ENC != "xt" else 2 * DD + 12, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, XI if ENC == "chan" else 2 * XI))
            if ENC != "chan":
                nn.init.zeros_(self.enc[-1].weight)
                nn.init.zeros_(self.enc[-1].bias)
        self.prior = Flow(0)
        self.head = Flow(WIDTH) if "head_once" in MODES else None
        self.register_buffer("log_lam", torch.tensor(math.log(LAM0)))
        self.register_buffer("P", torch.zeros(PCA_K, DD))
        self.register_buffer("P_mean", torch.zeros(DD))

    def embed(self, t):
        f = t[..., None] * self.freq
        return torch.cat([f.sin(), f.cos()], -1)

    def early(self, x, t, z=None):
        parts = [x.flatten(1), self.embed(t)] + ([z] if SITE == "in" else [])
        h = self.inp(torch.cat(parts, 1))
        for b in self.blocks[:SPLIT]:
            h = h + b(h)
        return h

    def late(self, h, z):
        if SITE == "late":
            h = h + self.xi(z)
        for b in self.blocks[SPLIT:]:
            h = h + b(h)
        return self.out(h)

    def encode(self, x, x_t, t):
        """z, KL in nats per sample."""
        if PCA_K:
            return (x.flatten(1) - self.P_mean) @ self.P.T, torch.zeros(len(x))
        if ENC == "chan":
            u = self.enc(x.flatten(1))
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-6) * XI ** 0.5
            return CHAN_A * u + (1 - CHAN_A ** 2) ** 0.5 * torch.randn_like(u), torch.zeros(len(x))
        q = self.enc(x.flatten(1)) if ENC == "x" else self.enc(torch.cat([x.flatten(1), x_t.flatten(1), self.embed(t)], 1))
        mu, ls = q.chunk(2, 1)
        return mu + (0.5 * ls).exp() * torch.randn_like(mu), 0.5 * (mu ** 2 + ls.exp() - ls - 1).sum(1)


def fit_pca(net):
    x = sample_data(20000).flatten(1)
    mean = x.mean(0)
    u, s, v = torch.linalg.svd(x - mean, full_matrices=False)
    k = min(PCA_K, (s > 1e-3 * s[0]).sum().item())
    p = v[:PCA_K] / (s[:PCA_K, None] / len(x) ** 0.5).clamp_min(1e-3)
    net.P.copy_(p)
    net.P_mean.copy_(mean)
    return k


def loss_fn(net, x):
    n = len(x)
    t = (torch.randn(n) * NORM[1] + NORM[0]).sigmoid()
    tv = t.view(-1, 1, 1)
    x_t = tv * x + (1 - tv) * torch.randn_like(x)
    w = 1 / (1 - t).clamp_min(T_EPS)
    z, kl = net.encode(x, x_t, t)
    h = net.early(x_t, t, z)
    pred = net.late(h, z).view(n, D, DT)
    loss = ((w.view(-1, 1, 1) * (pred - x)) ** 2).mean() + net.log_lam.exp() * kl.mean() / DD
    if TARGET is not None:
        with torch.no_grad():
            net.log_lam += 0.01 * (1.0 if kl.mean() > TARGET else -1.0)
            net.log_lam.clamp_(-7, 7)
    loss = loss + net.prior.loss(z.detach())
    if net.head is not None:
        # the head sees the features of x_t with stop-gradient; at t = 0 the input is noise, so it learns the prior; at low noise x_t fixes z
        loss = loss + net.head.loss(z.detach(), h.detach())
    return loss, kl.mean().detach()


def grid(calls, kind):
    if kind == "uniform":
        return torch.linspace(0, 1, calls + 1)
    q = torch.distributions.Normal(NORM[0], NORM[1]).icdf(torch.arange(1, calls) / calls).sigmoid()
    return torch.cat([torch.zeros(1), q, torch.ones(1)])


@torch.no_grad()
def generate(net, n, calls, mode, kind="uniform"):
    x = torch.randn(n, D, DT)
    ts = grid(calls, kind)
    if mode in ("gauss", "gauss_step"):
        z = torch.randn(n, XI)
    elif mode == "prior":
        z = net.prior.sample(n)
    elif mode == "head_once":
        z = net.head.sample(n, net.early(x, ts[0].expand(n)))
    else:
        z = None
    for i in range(calls):
        t, dt = ts[i].expand(n), ts[i + 1] - ts[i]
        if mode == "gauss_step":
            z = torch.randn(n, XI)
        if mode == "head_step":
            h = net.early(x, t)
            pred = net.late(h, net.head.sample(n, h)).view(n, D, DT)
            den = 1 - t
        else:
            pred = net.late(net.early(x, t, z), z).view(n, D, DT)
            den = (1 - t) if mode == "gauss_step" else (1 - t).clamp_min(T_EPS)
        x = x + dt * (pred - x) / den.view(-1, 1, 1)
    return x


def energy_distance(a, b):
    a, b = a.flatten(1), b.flatten(1)
    n, m = len(a), len(b)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).sum() / (n * (n - 1)) - torch.cdist(b, b).sum() / (m * (m - 1))).item()


def validity(x):
    if dataset in ("manifold", "embedded"):
        g = curve(torch.linspace(0, 2 * math.pi, 4000)[:, None]).flatten(1)
        d = torch.cdist(x.flatten(1), g)
        hist = torch.histc(d.argmin(1).float(), bins=64, min=0, max=3999) / len(x) * 64
        return {"dist": d.min(1).values.mean().item(), "coverage_min": hist.min().item(), "coverage_max": hist.max().item()}
    c = (CELLS + 0.5) / 2 - 1
    near = torch.cdist(x.flatten(0, 1), c).argmin(1).view(len(x), D)
    within = (x - c[near]).pow(2).mean().sqrt() / (0.5 / math.sqrt(12))
    return {"coherent": (near == near[:, :1]).all(1).float().mean().item(), "spread": within.item()}


if __name__ == "__main__":
    net = Net()
    if PCA_K:
        print("pca components with variance:", fit_pca(net), flush=True)
    ema = copy.deepcopy(net)
    params = [p for p in net.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0)
    rows, t_start = [], time.time()
    evals = {int(s / COST): s for s in (500, 1000, 2000, 5000, 10000, 20000) if s <= BUDGET}
    evals[int(BUDGET / COST)] = BUDGET
    for step in range(1, int(BUDGET / COST) + 1):
        loss, kl = loss_fn(net, sample_data(512))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            for p, q in zip(ema.parameters(), net.parameters()):
                p.lerp_(q, 1e-3)
            ema.log_lam.copy_(net.log_lam)
        if step in evals:
            real = sample_data(2000)
            row = {"step": evals[step], "real_steps": step, "dataset": dataset, "arm": arm, "seed": SEED, "seconds": time.time() - t_start, "kl": kl.item(), "lam": net.log_lam.exp().item(), "gen": []}
            for mode in MODES:
                for calls, kind in ((1, "uniform"), (2, "uniform"), (4, "uniform"), (8, "uniform"), (50, "uniform"), (8, "logit"), (50, "logit")):
                    g = generate(ema, 2000, calls, mode, kind)
                    row["gen"].append({"calls": calls, "grid": kind, "mode": mode, "ed": energy_distance(g, real), **validity(g)})
            rows.append(row)
            with open(out, "w") as f:
                json.dump(rows, f)
            print(dataset, arm, evals[step], step, f"{row['seconds']:.0f} s kl {row['kl']:.2f} lam {row['lam']:.3f}", " ".join(f"{g['mode'][:6]}{g['calls']}{g['grid'][0]}:{g['ed']:.3f}" for g in row["gen"] if g["calls"] in (1, 50)), flush=True)
