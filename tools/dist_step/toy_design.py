#!/usr/bin/env python3
"""Toy test of more ways to output a distribution at each step. CPU only. Data, backbone, recipe and compute accounting as in toy_implicit.py.

Arms. Compute: each arm trains for the same compute as the point arm's budget. COST charges extra blocks, the encoder and the flow head. Linear output layers
are not charged, since in the real model an output layer is 0.36% of a forward per image.
  point          the JiT recipe: x-prediction, v-space L2, Euler with the mean.
  impl_late_m2   as in toy_implicit.py: x_hat = g(x_t, t, xi), energy score, 2 samples per x_t, the last 2 blocks per sample. Sampled with xi fresh at each step (mode step)
                 and with one xi for the whole path (mode once).
  cvae_once_l<L> a code z ~ q(z | x), from a small encoder, joins before the last 2 blocks. Loss: v-space L2 plus L * KL / (output dims). Sampling: z ~ N(0, I) once per path, then
                 Euler. If the model is exact this is a flow conditioned on z, so the marginal is right at any number of calls.
  cvae_step_l<L> q(xi | x, x_t, t). Sampling: xi ~ N(0, I) fresh at each step, with the exact step.
  part_k<K>      K whole samples from one forward, whole-sample energy score, one forward per x_t. Sampling modes: step (a random particle at each step), once (one particle index for the
                 path), mean (the mean of the K, Euler).
  gauss_r0       a Gaussian prior per token: mean and one variance. x_hat is its closed-form posterior mean, trained with the point loss. Modes: mean (Euler), sample (an exact posterior
                 sample by Matheron's rule, the exact step).
  gauss_r0d      the same with one variance per token dimension.
  gauss_r<R>     a Gaussian prior on the whole sample: mean, a variance per token and R directions over all tokens.
  nll_d          mean with the point loss, plus a Gaussian residual r = (x - m) / (1 - t) with a variance per token dimension, negative log-likelihood, stop-gradient on m. Modes: mean, sample.
  nll_r<R>       the same with R directions over all tokens (posterior principal components).
  flow_w48       mean with the point loss, plus a small flow per token on the same residual, conditioned on trunk features of that token. Modes: mean, sample.

usage: toy_design.py <dataset> <arm> <out.json> [compute in point-arm steps] [seed]
"""
import copy
import json
import math
import re
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
T_EPS, XI, Z, HW = 0.05, 32, 32, 48
WIDTH, DEPTH, SPLIT = 256, 4, 2
NORM_LOW = -0.8, 0.8

KIND = arm.split("_")[0]
M = 2 if KIND == "impl" else 1
K = int(arm.split("_k")[1]) if KIND == "part" else 1
LAM = float(arm.split("_l")[1]) if KIND == "cvae" else 0.0
ONCE = KIND == "cvae" and arm.startswith("cvae_once")
m_ = re.match(r"(gauss|nll)_(r(\d+)|d)(d?)$", arm)
R = int(m_.group(3) or 0) if m_ else 0
DIAG = bool(m_) and (m_.group(4) == "d" or m_.group(2) == "d")
COST = {"impl": 1.5, "cvae": 1.04, "flow": 1.17}.get(KIND, 1.0)
MODES = {"point": ["mean"], "impl": ["step", "once"], "cvae": ["once" if ONCE else "step"], "part": ["step", "once", "mean"], "gauss": ["mean", "sample"], "nll": ["mean", "sample"], "flow": ["mean", "sample"]}[KIND]

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


def n_out():
    if KIND in ("point", "impl", "cvae"):
        return DD
    if KIND == "part":
        return K * DD
    if KIND == "gauss":
        return DD + (DD if DIAG else D) + R * DD
    if KIND == "nll":
        return DD + DD + R * DD
    return DD + D * Z                                           # flow: the mean and features per token


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(6) * math.pi)
        self.inp = nn.Linear(DD + 12, WIDTH)
        self.blocks = nn.ModuleList(block(WIDTH) for _ in range(DEPTH))
        self.out = nn.Linear(WIDTH, n_out())
        if KIND in ("impl", "cvae"):
            self.xi = nn.Linear(XI, WIDTH)
        if KIND == "cvae":
            self.enc = nn.Sequential(nn.Linear(DD if ONCE else 2 * DD + 12, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2 * XI))
            nn.init.zeros_(self.enc[-1].weight)
            nn.init.zeros_(self.enc[-1].bias)
        if KIND == "flow":
            self.h_in = nn.Linear(DT + 12 + Z, HW)
            self.h_blocks = nn.ModuleList(block(HW) for _ in range(2))
            self.h_out = nn.Linear(HW, DT)

    def embed(self, t):
        f = t[..., None] * self.freq
        return torch.cat([f.sin(), f.cos()], -1)

    def trunk(self, x, t, xi=None):
        """xi: (n, XI), or (n, m, XI) for m samples per input."""
        h = self.inp(torch.cat([x.flatten(1), self.embed(t)], 1))
        for b in self.blocks[:SPLIT]:
            h = h + b(h)
        if xi is not None:
            if xi.dim() == 3:
                h = h.repeat_interleave(xi.shape[1], 0)
                xi = xi.flatten(0, 1)
            h = h + self.xi(xi)
        for b in self.blocks[SPLIT:]:
            h = h + b(h)
        return self.out(h)

    def head(self, y, tau, z):
        """The flow head for one token: y (.., DT), tau (..), z (.., Z)."""
        h = self.h_in(torch.cat([y, self.embed(tau), z], -1))
        for b in self.h_blocks:
            h = h + b(h)
        return self.h_out(h)


def gauss_parts(o):
    mu = o[:, :DD]
    if DIAG:
        s2, rest = o[:, DD:2 * DD], o[:, 2 * DD:]
    else:
        s2, rest = o[:, DD:DD + D].repeat_interleave(DT, 1), o[:, DD + D:]
    return mu, s2.exp() + 1e-6, rest.view(-1, DD, R) if R else None


def sigma_times_inverse(s2, U, a, b, r):
    """Sigma (a^2 Sigma + b^2 I)^-1 r, Sigma = diag(s2) + U U^T, by the Woodbury identity. a, b: (n, 1). r: (n, DD)."""
    d = a ** 2 * s2 + b ** 2
    y = r / d
    if U is None:
        return s2 * y
    Ud = U / d[..., None]
    Ut = U.transpose(1, 2)
    c = torch.linalg.solve(torch.eye(R) + a[..., None] ** 2 * (Ut @ Ud), Ut @ y[..., None])
    y = y - a ** 2 * (Ud @ c)[..., 0]
    return s2 * y + (U @ (Ut @ y[..., None]))[..., 0]


def gauss_mean(o, x, t):
    mu, s2, U = gauss_parts(o)
    a, b = t[:, None], (1 - t)[:, None]
    return (mu + a * sigma_times_inverse(s2, U, a, b, x.flatten(1) - a * mu)).view(-1, D, DT)


def gauss_sample(o, x, t):
    """Matheron's rule: a prior sample, corrected by the difference between the observation and a simulated observation."""
    mu, s2, U = gauss_parts(o)
    a, b = t[:, None], (1 - t)[:, None]
    prior = mu + s2.sqrt() * torch.randn_like(mu)
    if U is not None:
        prior = prior + (U @ torch.randn(len(mu), R, 1))[..., 0]
    sim = a * prior + b * torch.randn_like(mu)
    return (prior + a * sigma_times_inverse(s2, U, a, b, x.flatten(1) - sim)).view(-1, D, DT)


def nll_parts(o):
    m, ls = o[:, :DD], o[:, DD:2 * DD]
    return m, ls, o[:, 2 * DD:].view(-1, DD, R) if R else None


def loss_fn(net, x):
    n = len(x)
    t = (torch.randn(n) * NORM_LOW[1] + NORM_LOW[0]).sigmoid()
    tv = t.view(-1, 1, 1)
    eps = torch.randn_like(x)
    x_t = tv * x + (1 - tv) * eps
    w = 1 / (1 - t).clamp_min(T_EPS)
    wv = w.view(-1, 1, 1)
    if KIND == "point":
        return ((wv * (net.trunk(x_t, t).view(n, D, DT) - x)) ** 2).mean()
    if KIND == "gauss":
        return ((wv * (gauss_mean(net.trunk(x_t, t), x_t, t) - x)) ** 2).mean()
    if KIND == "cvae":
        q = net.enc(x.flatten(1)) if ONCE else net.enc(torch.cat([x.flatten(1), x_t.flatten(1), net.embed(t)], 1))
        mu, ls = q.chunk(2, 1)
        xi = mu + (0.5 * ls).exp() * torch.randn_like(mu)
        kl = 0.5 * (mu ** 2 + ls.exp() - ls - 1).sum(1)
        return ((wv * (net.trunk(x_t, t, xi).view(n, D, DT) - x)) ** 2).mean() + LAM * kl.mean() / DD
    if KIND in ("impl", "part"):
        if KIND == "impl":
            g = net.trunk(x_t, t, torch.randn(n, M, XI)).view(n, M, D, DT)
        else:
            g = net.trunk(x_t, t).view(n, K, D, DT)
        gf, xf, c = g.flatten(2), x.flatten(1)[:, None], g.shape[1]
        return (w * ((gf - xf).norm(dim=-1).mean(1) - 0.5 * torch.cdist(gf, gf).sum((1, 2)) / (c * (c - 1)))).mean()
    o = net.trunk(x_t, t)
    m = o[:, :DD]
    point = ((w[:, None] * (m - x.flatten(1))) ** 2).mean()
    r = ((x.flatten(1) - m.detach()) * w[:, None]).clamp(-50, 50)   # the residual of the velocity
    if KIND == "nll":
        _, ls, U = nll_parts(o)
        ls = ls.clamp(-12, 8)
        s2 = ls.exp()
        if U is None:
            return point + 0.5 * ((r ** 2 / s2 + ls).sum(1)).mean() / DD
        Ud = U / s2[..., None]
        Ut = U.transpose(1, 2)
        I_ = torch.eye(R) + Ut @ Ud
        rs = r / s2
        c = torch.linalg.solve(I_, Ut @ rs[..., None])
        quad = (r * rs).sum(1) - ((Ut @ rs[..., None]) * c).sum((1, 2))
        return point + 0.5 * ((quad + ls.sum(1) + torch.linalg.slogdet(I_)[1])).mean() / DD
    z = o[:, DD:].view(n, D, Z)                                 # flow: a small flow per token on the residual
    tau = torch.rand(n, D)
    eta = torch.randn(n, D, DT)
    y = tau[..., None] * r.view(n, D, DT) + (1 - tau[..., None]) * eta
    return point + ((net.head(y, tau, z) - (r.view(n, D, DT) - eta)) ** 2).mean()


def grid(calls, kind):
    if kind == "uniform":
        return torch.linspace(0, 1, calls + 1)
    u = (torch.arange(1, calls) / calls)                        # the training density of t: interior points at its quantiles
    q = torch.distributions.Normal(NORM_LOW[0], NORM_LOW[1]).icdf(u).sigmoid()
    return torch.cat([torch.zeros(1), q, torch.ones(1)])


@torch.no_grad()
def generate(net, n, calls, mode, kind="uniform"):
    x = torch.randn(n, D, DT)
    ts = grid(calls, kind)
    z0 = torch.randn(n, XI)
    k0 = torch.randint(K, (n,))
    for i in range(calls):
        t, dt = ts[i].expand(n), ts[i + 1] - ts[i]
        euler = mode == "mean" or (KIND == "cvae" and ONCE)
        if KIND == "point":
            pred = net.trunk(x, t).view(n, D, DT)
        elif KIND in ("impl", "cvae"):
            pred = net.trunk(x, t, z0 if mode == "once" else torch.randn(n, XI)).view(n, D, DT)
        elif KIND == "part":
            g = net.trunk(x, t).view(n, K, D, DT)
            if mode == "mean":
                pred = g.mean(1)
            else:
                k = k0 if mode == "once" else torch.randint(K, (n,))
                pred = g[torch.arange(n), k]
        elif KIND == "gauss":
            o = net.trunk(x, t)
            pred = gauss_mean(o, x, t) if mode == "mean" else gauss_sample(o, x, t)
        elif KIND == "nll":
            o = net.trunk(x, t)
            m, ls, U = nll_parts(o)
            r = torch.zeros_like(m)
            if mode == "sample":
                r = (0.5 * ls.clamp(-12, 8)).exp() * torch.randn_like(m)
                if U is not None:
                    r = r + (U @ torch.randn(n, R, 1))[..., 0]
            pred = (m + (1 - t)[:, None] * r).view(n, D, DT)
        else:
            o = net.trunk(x, t)
            m, zf = o[:, :DD].view(n, D, DT), o[:, DD:].view(n, D, Z)
            if mode == "mean":
                pred = m
            else:
                y = torch.randn(n, D, DT)
                for j in range(8):
                    y = y + net.head(y, torch.full((n, D), j / 8), zf) / 8
                pred = m + (1 - t).view(-1, 1, 1) * y
        den = (1 - t).clamp_min(T_EPS) if euler else (1 - t)
        x = x + dt * (pred - x) / den.view(-1, 1, 1)
    return x


def energy_distance(a, b):
    a, b = a.flatten(1), b.flatten(1)
    n, m = len(a), len(b)
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).sum() / (n * (n - 1)) - torch.cdist(b, b).sum() / (m * (m - 1))).item()


def validity(x):
    if dataset in ("manifold", "embedded"):
        ths = torch.linspace(0, 2 * math.pi, 4000)
        g = curve(ths[:, None]).flatten(1)
        d = torch.cdist(x.flatten(1), g)
        hist = torch.histc(d.argmin(1).float(), bins=64, min=0, max=3999) / len(x) * 64
        return {"dist": d.min(1).values.mean().item(), "coverage_min": hist.min().item(), "coverage_max": hist.max().item()}
    c = (CELLS + 0.5) / 2 - 1
    near = torch.cdist(x.flatten(0, 1), c).argmin(1).view(len(x), D)
    within = (x - c[near]).pow(2).mean().sqrt() / (0.5 / math.sqrt(12))
    share = torch.bincount(near.flatten(), minlength=8).float() / near.numel() * 8
    return {"coherent": (near == near[:, :1]).all(1).float().mean().item(), "spread": within.item(), "share_min": share.min().item(), "share_max": share.max().item()}


@torch.no_grad()
def kl_now(net, n=2000):
    if KIND != "cvae":
        return None
    x = sample_data(n)
    t = (torch.randn(n) * NORM_LOW[1] + NORM_LOW[0]).sigmoid()
    x_t = t.view(-1, 1, 1) * x + (1 - t).view(-1, 1, 1) * torch.randn_like(x)
    q = net.enc(x.flatten(1)) if ONCE else net.enc(torch.cat([x.flatten(1), x_t.flatten(1), net.embed(t)], 1))
    mu, ls = q.chunk(2, 1)
    return (0.5 * (mu ** 2 + ls.exp() - ls - 1).sum(1)).mean().item()


if __name__ == "__main__":
    net = Net()
    ema = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
    rows, t_start = [], time.time()
    evals = {int(s / COST): s for s in (500, 1000, 2000, 5000, 10000, 20000) if s <= BUDGET}
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
            row = {"step": evals[step], "real_steps": step, "dataset": dataset, "arm": arm, "seed": SEED, "seconds": time.time() - t_start, "kl": kl_now(ema), "gen": []}
            for mode in MODES:
                for calls, kind in ((1, "uniform"), (2, "uniform"), (4, "uniform"), (8, "uniform"), (50, "uniform"), (8, "logit"), (50, "logit")):
                    g = generate(ema, 2000, calls, mode, kind)
                    row["gen"].append({"calls": calls, "grid": kind, "mode": mode, "ed": energy_distance(g, real), **validity(g)})
            rows.append(row)
            with open(out, "w") as f:
                json.dump(rows, f)
            print(dataset, arm, evals[step], step, f"{row['seconds']:.0f} s", " ".join(f"{g['mode'][:2]}{g['calls']}{g['grid'][0]}:{g['ed']:.3f}" for g in row["gen"]), flush=True)
