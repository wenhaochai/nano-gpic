"""FLOP-matched steps of the distributional denoiser on the final ladder (docs/ddm.md), as tools/gmflow_ladder/steps.py does for the mixture.

The point model's forward FLOPs per image are the ladder's own (tmp/ladder_prof/all_points.py fwd(), with the mean caption length; adaLN and elementwise
work are not counted there, nor here). The distributional denoiser (configs_h100/ddm_train.yaml) adds, per image, with m = 4 samples, c = 256 noise channels,
L = 256 image tokens, T = caption tokens, P = 768 values per token:
  - the last 2 blocks run m times instead of once: 2 (m - 1) (32 L d^2 + 4 T d^2 + 4 L (L + T) d), fwd()'s block;
  - in those blocks the residual stream is d + c wide: the projections into and out of attention and the MLP read or write c more channels,
    32 L d c more per block and sample: 2 m 32 L d c;
  - the final layer runs m times: (m - 1) 2 L d P.
r = added / fwd(); steps = point steps / (1 + r), rounded down to a multiple of 32; Muon decay = 5 / (Muon LR x steps), the sweep's rule.
tools/ddm_ladder/test_ddm.py checks r against torch's FlopCounterMode on the real models. usage: python -I steps.py   (a pilot CPU or a vis node)"""
WC = "/scratch/gpfs/GROUP/USER"
src = open(f"{WC}/tmp/ladder_prof/all_points.py").read()
exec(src[src.index("import glob"):src.index("import yaml")])
exec(src[src.index("WC = "):src.index("def params_formula")])   # T_hist -> ET, ET2; L, P, B; fwd()

M, C = 4, 256
# tag, d, nb, nt, steps, Muon LR, AdamW LR, Muon decay, data, batch, cooldown % (tmp/ladder_prof/wsd/sweep_list_xl3.txt, as tools/gmflow_ladder/steps.py)
POINT = [("xl-d10-0p25c0", 768, 10, 2, 21248, "1.360e-02", "0.000501", "0.0173", "full", 1024, 50),
         ("xl-d12-1c0", 1024, 12, 3, 40192, "6.800e-03", "0.000326", "0.01829", "full", 1024, 50)]


def added(d, T, m=M, c=C, attn=True):
    block = 32 * L * d * d + 4 * T * d * d + (4 * L * (L + T) * d if attn else 0)
    return 2 * (m - 1) * block + 2 * m * 32 * L * d * c + (m - 1) * 2 * L * d * P


def fwd_no_attn(d, nb, nt, T, T2):
    """fwd() without the attention products (q k^T and the weighted sum), which torch's FlopCounterMode does not count for SDPA on a CPU."""
    return fwd(d, nb, nt, T, T2) - nb * 4 * L * (L + T) * d - nt * 4 * T2 * d


def ratio(d, nb, nt, T=None, T2=None):
    T = ET if T is None else T
    T2 = ET2 if T2 is None else T2
    return added(d, T) / fwd(d, nb, nt, T, T2)


if __name__ == "__main__":
    rows = []
    for tag, d, nb, nt, S, mlr, alr, wdk, data, gb, cdp in POINT:
        f = fwd(d, nb, nt, ET, ET2)
        r = ratio(d, nb, nt)
        s = int(S / (1 + r)) // 32 * 32
        print(f"{tag}: forward {f / 1e9:.3f} GFLOP per image (mean caption {ET:.2f} tokens), point FLOPs {3 * f * 1024 * S:.4g}; ddm adds {added(d, ET) / 1e9:.3f} GFLOP,"
              f" r {r:.5f}, steps {s}, s (1 + r) {s * (1 + r):.0f} <= {S}, budget used {s * (1 + r) / S:.5f}, FLOPs {3 * f * (1 + r) * 1024 * s:.4g}")
        rows.append(f"{tag.replace('xl-', 'ddm-')} {d} {nb} {nt} {s} {mlr} {alr} {5 / (float(mlr) * s):.4g} {data} {gb} {cdp}")
    print("\n".join(rows))
