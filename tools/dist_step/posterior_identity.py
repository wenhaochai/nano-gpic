#!/usr/bin/env python3
"""Numeric check of the three identities the design note uses, on a Gaussian mixture prior in d dimensions.
x_sigma = x + sigma * eps (variance-exploding form; any alpha, sigma path maps to it by x_t / alpha_t, sigma_t / alpha_t).
1. tilt: the posterior at sigma_s < sigma_t is the posterior at sigma_t times a Gaussian ratio.
2. ODE derivative: dD/dsigma along the probability flow ODE = (m3 - Cov * r) / sigma^3, r = y - D, m3 = E[e |e|^2], e = x - D.
3. All velocities on the path come from one posterior: integrating with the tilted posterior of the first point gives the exact path."""
import numpy as np
rng = np.random.default_rng(0)
d, K = 16, 8
mu = rng.normal(size=(K, d)) * 2.0
pi = rng.dirichlet(np.ones(K))
s2 = rng.uniform(0.05, 0.3, size=K)              # component variances of the prior

def post(y, sig):
    """Posterior of x | y at sigma sig: (weights, means, variances) of an isotropic mixture."""
    v = s2 + sig ** 2
    logw = np.log(pi) - 0.5 * ((y - mu) ** 2).sum(1) / v - 0.5 * d * np.log(v)
    w = np.exp(logw - logw.max()); w /= w.sum()
    g = s2 / v
    m = mu + g[:, None] * (y - mu)
    return w, m, g * sig ** 2

def moments(w, m, c):
    D = (w[:, None] * m).sum(0)
    e = m - D
    Cov = (w[:, None, None] * (e[:, :, None] * e[:, None, :] + c[:, None, None] * np.eye(d))).sum(0)
    # E[e |e|^2] for component k: mean e_k, isotropic variance c_k: e_k (|e_k|^2 + (d + 2) c_k)
    m3 = (w[:, None] * e * ((e ** 2).sum(1) + (d + 2) * c)[:, None]).sum(0)
    return D, Cov, m3

def tilt(w, m, c, y_t, sig_t, y_s, sig_s):
    """Posterior at (y_s, sig_s) from the posterior at (y_t, sig_t) only: multiply by N(y_s; x, sig_s^2) / N(y_t; x, sig_t^2)."""
    lam = 1 / c + 1 / sig_s ** 2 - 1 / sig_t ** 2                       # precision of the new component
    cn = 1 / lam
    mn = cn[:, None] * (m / c[:, None] + y_s / sig_s ** 2 - y_t / sig_t ** 2)
    # log of the integral of N(x; m, c) N(y_s; x, sig_s^2) / N(y_t; x, sig_t^2) dx, up to terms free of k
    logz = 0.5 * d * np.log(cn / c) + 0.5 * ((mn ** 2).sum(1) / cn - (m ** 2).sum(1) / c)
    logw = np.log(w) + logz
    wn = np.exp(logw - logw.max()); wn /= wn.sum()
    return wn, mn, cn

x = mu[3] + np.sqrt(s2[3]) * rng.normal(size=d)
sig_t, sig_s = 3.0, 0.7
y_t = x + sig_t * rng.normal(size=d)
y_s = x + sig_s * rng.normal(size=d)
wt, mt, ct = post(y_t, sig_t)
we, me, ce = post(y_s, sig_s)
wn, mn, cn = tilt(wt, mt, ct, y_t, sig_t, y_s, sig_s)
print("1. tilt vs exact posterior at sigma_s:  max weight error %.2e, max mean error %.2e, max variance error %.2e" % (np.abs(wn - we).max(), np.abs(mn - me).max(), np.abs(cn - ce).max()))

def D_of(y, sig):
    return moments(*post(y, sig))[0]

y = y_t.copy(); sig = sig_t
D, Cov, m3 = moments(*post(y, sig))
h = 1e-5
y2 = y + h * (y - D) / sig          # one Euler step of dy/dsigma = (y - D) / sigma
num = (D_of(y2, sig + h) - D) / h
ana = (m3 - Cov @ (y - D)) / sig ** 3
print("2. dD/dsigma along the ODE: analytic vs finite difference, relative error %.2e" % (np.linalg.norm(num - ana) / np.linalg.norm(num)))

def path(y0, sig0, sig1, steps, field):
    y = y0.copy()
    sg = np.geomspace(sig0, sig1, steps + 1)
    for a, b in zip(sg, sg[1:]):
        k1 = (y - field(y, a)) / a
        ym = y + (b - a) / 2 * k1; c = np.sqrt(a * b) if False else (a + b) / 2
        k2 = (ym - field(ym, c)) / c
        y = y + (b - a) * k2
    return y

ref = path(y_t, sig_t, 0.05, 4000, D_of)
free = lambda yy, sg: (lambda w, m, c: (w[:, None] * m).sum(0))(*tilt(wt, mt, ct, y_t, sig_t, yy, sg)) if sg < sig_t - 1e-12 else D_of(yy, sg)
one = path(y_t, sig_t, 0.05, 400, free)
print("3. whole path from one posterior, 400 midpoint steps: distance to the exact path end %.2e (path length scale %.2f)" % (np.linalg.norm(one - ref), np.linalg.norm(ref - y_t)))
print("   point output, N Euler steps from sigma=3 to 0.05 on the exact field; distance to the exact path end:")
for n in (1, 2, 5, 10, 50):
    print("    N=%3d  %.3f" % (n, np.linalg.norm(path(y_t, sig_t, 0.05, n, D_of) - ref)))
