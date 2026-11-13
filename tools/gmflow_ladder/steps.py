"""FLOP-matched steps of the per-pixel mixture on the final ladder (docs/gmflow_ladder_port.md; docs/gmflow.md 4.2; tmp/ladder_prof/gm_full_steps.py).

r = the head's added forward FLOPs / the point model's forward FLOPs per image at that shape. The added FLOPs are the final linear's wider output,
2 L d (K (C + 1) p^2 - C p^2) = 2 * 256 * d * (K * 1024 - 768), plus the s-MLP, 2 (d * 1024 + 1024) (tools/gmflow_ladder/test_port.py measures both
with FlopCounterMode). The forward FLOPs per image are the ladder's own (tmp/ladder_prof/all_points.py fwd() with the mean caption length, the
function tmp/ladder_prof/wsd/ladder_plan.py uses for the sweep's FLOPs). Steps = point steps / (1 + r), rounded down to a multiple of 32.
Muon decay = 5 / (Muon LR x steps), the sweep's rule. usage: python -I steps.py   (vis node)"""
WC = "/scratch/gpfs/GROUP/USER"
src = open(f"{WC}/tmp/ladder_prof/all_points.py").read()
exec(src[src.index("import glob"):src.index("import yaml")])
exec(src[src.index("WC = "):src.index("def params_formula")])   # T_hist -> ET, ET2; L, P, B; fwd()

# tag, d, nb, nt, steps, Muon LR, AdamW LR, Muon decay, data, batch, cooldown % (tmp/ladder_prof/wsd/sweep_list_xl3.txt, sweep_list_xl2.txt)
POINT = [("xl-d10-0p25c0", 768, 10, 2, 21248, "1.360e-02", "0.000501", "0.0173", "full", 1024, 50),
         ("xl-d12-1c0", 1024, 12, 3, 40192, "6.800e-03", "0.000326", "0.01829", "full", 1024, 50),
         ("xl-d14-4c0", 1280, 14, 3, 89600, "3.400e-03", "0.000238", "0.01641", "full", 1024, 50)]   # third budget, 5.7e19 FLOPs (owner popup 2026-11-12 22:5x); FSDP2


def added(d, K):
    return 2 * 256 * d * (K * 1024 - 768) + 2 * (d * 1024 + 1024)


rows = []
for tag, d, nb, nt, S, mlr, alr, wdk, data, gb, cdp in POINT:
    f = fwd(d, nb, nt, ET, ET2)
    print(f"{tag}: forward {f / 1e9:.3f} GFLOP per image (mean caption {ET:.2f} tokens), point FLOPs {3 * f * 1024 * S:.4g}, decay check {5 / (float(mlr) * S):.5g} (list {wdk})")
    for K in (8, 1):
        r = added(d, K) / f
        s = int(S / (1 + r)) // 32 * 32
        name = tag.replace("xl-", f"gm{K}-")
        print(f"  K={K}: added {added(d, K) / 1e6:.2f} MFLOP, r {r:.5f}, steps {s}, s (1 + r) {s * (1 + r):.0f} <= {S}, budget used {s * (1 + r) / S:.5f},"
              f" FLOPs {3 * f * (1 + r) * 1024 * s:.4g}")
        rows.append(f"{name} {d} {nb} {nt} {s} {mlr} {alr} {5 / (float(mlr) * s):.4g} {data} {gb} {cdp}")
print("\n".join(rows))
