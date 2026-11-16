#!/usr/bin/env python3
"""The scaling law of the post "How to Scale Text-to-image Models: a Handbook" (section "Our Scaling Law"), from its 38 runs.

One formula for the held-out loss of a model with N image parameters trained on D image tokens:
    L(N, D) = E + (A / N^alpha + B / D^beta)^k          (Chinchilla's form when k = 1)
N = 18 x image layers x width^2 (the image layers' matrices; text layers, adaLN and embeddings left out), D = image tokens seen
(256 per image), C = 6 N D. The fit minimizes the weighted squared relative error over the runs (least squares, 48 random starts, the
best kept), in the variables x = ln(N / 1e8), y = ln(D / 1e10): L = E + K [e^(-alpha x) + R e^(-beta y)]^k, which gives A and B back.
Each run's weight is exp(-d^2 / (2 x 0.4^2)) x sqrt(C / 3.6e18), d = ln N minus the vertex of its budget's iso-FLOP parabola (a quadratic
in ln N through that budget's runs): runs far from a minimum pull the optimum off (equal weights: N* ~ C^0.515; only the runs within 2x
of a minimum: C^0.50), and larger budgets are closer to where the law is extrapolated. --equal-weights fits every run alike.
For a budget C the loss is lowest at N* = G (C / 6)^(beta / (alpha + beta)), D* = C / (6 N*), G = (alpha A / (beta B))^(1 / (alpha + beta))
(Weng 2026's closed form; k does not move the optimum). --boot resamples the runs (each keeps its weight) for 90% intervals.
At each optimum it also prints the post's recipe: Muon lr = 1.36e-2 (C / 3.567e18)^-0.5, AdamW lr = 6.4e-4 (all params / 109.4M)^-0.547,
Muon weight decay = 5 / (Muon lr x steps), steps = D* / (1024 x 256); all params from N* by a power law over the runs' shapes.

Data: final_ladder_runs.csv beside this file: every run of the iso-FLOP curves with the post's recipe (50% cooldown, the rate rule,
batch 1024, GPIC Full), 3.6e18 to 2.3e20 FLOPs; heldout_loss is tools/heldout_loss.py's mean over 16 noise levels on 3,072 held-out
images, raw weights at the last step; all_params counts every parameter of the model.
usage: python tools/fit/scaling_law.py [--csv final_ladder_runs.csv] [--boot 200] [--equal-weights]   (numpy and scipy; a minute with --boot)
Expected (the post): E 0.0457, alpha 0.52, beta 0.52, k 0.20, N* ~ C^0.50, 50 to 51 tokens per parameter, a loss error of 0.057%
(0.025% for the runs within 2x of a minimum); at the optimum a Muon weight decay of about 0.0175 at every budget.
"""
import argparse, csv, math, os
import numpy as np
from scipy.optimize import least_squares

#        name   low   high  range of the random starts
SPEC = [("E", 0.0, 0.0645, (0.055, 0.064)), ("lnK", -25, 5, (-9, -3)), ("alpha", 0.005, 3, (0.05, 1.5)), ("beta", 0.005, 3, (0.05, 1.5)),
        ("lnR", -15, 15, (-3, 3)), ("k", 0.005, 3, (0.05, 1.2))]
LO, HI = np.array([s[1] for s in SPEC]), np.array([s[2] for s in SPEC])
C0 = 3.567e18   # the smallest budget (d10); budgets are C0 x 2^j


def predict(t, x, y):
    E, lnK, a, b, lnR, k = t
    return E + np.exp(lnK + k * np.logaddexp(-a * x, lnR - b * y))


def fit(x, y, L, w, nstart, seed, extra=()):
    rng = np.random.default_rng(seed)
    best = (None, float("inf"))
    sw = np.sqrt(w)
    for s in list(extra) + [np.array([rng.uniform(*sp[3]) for sp in SPEC]) for _ in range(nstart)]:
        s = np.clip(s, LO + 1e-9 * (HI - LO), HI - 1e-9 * (HI - LO))
        try:
            with np.errstate(all="ignore"):
                r = least_squares(lambda t: sw * (predict(t, x, y) - L) / L, s, bounds=(LO, HI), method="trf", x_scale="jac",
                                  max_nfev=4000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if np.isfinite(rss) and rss < best[1]:
            best = (r.x, rss)
    return best[0]


def vertex_distance(N, L, C):
    """d = ln N minus the vertex of the run's budget's parabola (L quadratic in ln N over the runs of that budget)."""
    b = np.round(np.log2(C / C0)).astype(int)
    d = np.zeros(len(N))
    for j in set(b):
        i = b == j
        p = np.polyfit(np.log(N[i]), L[i], 2)
        d[i] = np.log(N[i]) + p[1] / (2 * p[0])
    return d


def constants(t):
    E, lnK, a, b, lnR, k = t
    K1 = math.exp(lnK / k)
    return dict(E=E, A=K1 * 1e8 ** a, B=K1 * math.exp(lnR) * 1e10 ** b, alpha=a, beta=b, k=k)


def optimum(p, C):
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1 / (p["alpha"] + p["beta"]))
    N = G * (C / 6) ** (p["beta"] / (p["alpha"] + p["beta"]))
    return N, C / (6 * N)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_ladder_runs.csv"))
    ap.add_argument("--boot", type=int, default=0, help="bootstrap resamples of the runs for 90%% intervals (the post: 200)")
    ap.add_argument("--equal-weights", action="store_true", help="weight every run alike (the post's weights otherwise)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    N = np.array([float(r["image_params"]) for r in rows]); D = np.array([float(r["image_tokens"]) for r in rows])
    L = np.array([float(r["heldout_loss"]) for r in rows]); C = np.array([float(r["flops"]) for r in rows])
    x, y = np.log(N / 1e8), np.log(D / 1e10)
    d = vertex_distance(N, L, C)
    w = np.ones(len(rows)) if a.equal_weights else np.exp(-d ** 2 / (2 * 0.4 ** 2)) * np.sqrt(C / C0)
    t = fit(x, y, L, w, 48, 1)
    p = constants(t)
    err = predict(t, x, y) / L - 1
    near = np.abs(d) <= math.log(2)
    eN = p["beta"] / (p["alpha"] + p["beta"])
    print(f"{len(rows)} runs ({'equal weights' if a.equal_weights else 'weighted to each minimum and to compute'}); L = E + (A/N^alpha + B/D^beta)^k")
    print("  " + ", ".join(f"{k} {v:.4g}" for k, v in p.items()))
    print(f"  loss error: RMS {100 * np.sqrt(np.mean(err ** 2)):.3f}% ({100 * np.sqrt(np.mean(err[near] ** 2)):.3f}% for the {near.sum()} runs within 2x "
          f"of a minimum), largest {100 * np.max(np.abs(err)):.3f}%")
    print(f"  N* ~ C^{eN:.3f}, D* ~ C^{1 - eN:.3f}")
    shapes = {(r["width"], r["image_layers"], r["text_layers"]): (float(r["image_params"]), float(r["all_params"])) for r in rows}
    g, c = np.polyfit(*np.log(np.array(list(shapes.values()))).T, 1)   # ln(all params) = c + g ln(N)
    print("  budget   N* (image parameters)  D* (image tokens)  tokens per parameter  predicted loss  Muon lr  AdamW lr  Muon weight decay")
    for Cb in [C0 * 2 ** j for j in range(7)] + [1e21]:
        n, dd = optimum(p, Cb)
        mlr = 1.36e-2 * (Cb / C0) ** -0.5
        alr = 6.4e-4 * (math.exp(c + g * math.log(n)) / 109.4e6) ** -0.547
        print(f"  {Cb:.2g}  {n / 1e6:12.0f}M  {dd / 1e9:15.1f}B  {dd / n:17.1f}  {p['E'] + (p['A'] / n ** p['alpha'] + p['B'] / dd ** p['beta']) ** p['k']:15.5f}"
              f"  {mlr:.2e}  {alr:.2e}  {5 / (mlr * dd / (1024 * 256)):.4f}")
    if a.boot:
        rng = np.random.default_rng(7)
        eNs, tpp = [], []
        for _ in range(a.boot):
            i = rng.choice(len(rows), len(rows))
            tb = fit(x[i], y[i], L[i], w[i], 4, int(rng.integers(1e9)), extra=[t])
            pb = constants(tb)
            eNs.append(pb["beta"] / (pb["alpha"] + pb["beta"]))
            n, dd = optimum(pb, 2.3e20)
            tpp.append(dd / n)
        lo, hi = np.percentile(eNs, [5, 95])
        print(f"  {a.boot} bootstrap resamples: N* exponent 90% interval [{lo:.3f}, {hi:.3f}];"
              f" tokens per parameter at 2.3e20 [{np.percentile(tpp, 5):.1f}, {np.percentile(tpp, 95):.1f}]")


if __name__ == "__main__":
    main()
