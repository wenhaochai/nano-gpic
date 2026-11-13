"""FLOP-matched steps of the DTM flow head on the final ladder (docs/dtm_ladder.md; as tools/gmflow_ladder/steps.py for the mixture).

r = (the head's forward FLOPs per image - the point model's final linear, which the head replaces) / the point model's forward FLOPs per image at
that shape (tmp/ladder_prof/all_points.py fwd() with the mean caption length). The head runs once per patch per training image (one head time per
patch), so the training FLOPs per image are 3 x forward x (1 + r). Steps = point steps / (1 + r), rounded down to a multiple of 32; Muon decay =
5 / (Muon LR x steps), the sweep's rule. The head's FLOPs are counted with FlopCounterMode on the model's own TokenFlowHead (2 per multiply-add).
usage (vis node): tools/gmflow_ladder/vis.sh tools/dtm_ladder/steps.py"""
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.transformer.JiT_T2I import TokenFlowHead

WC = "/scratch/gpfs/GROUP/USER"
src = open(f"{WC}/tmp/ladder_prof/all_points.py").read()
exec(src[src.index("import glob"):src.index("import yaml")])
exec(src[src.index("WC = "):src.index("def params_formula")])   # T_hist -> ET, ET2; L, P, B; fwd()

# tag, d, nb, nt, steps, Muon LR, AdamW LR, Muon decay, data, batch, cooldown % (tmp/ladder_prof/wsd/sweep_list_xl3.txt, sweep_list_xl2.txt)
POINT = [("xl-d10-0p25c0", 768, 10, 2, 21248, "1.360e-02", "0.000501", "0.0173", "full", 1024, 50),
         ("xl-d12-1c0", 1024, 12, 3, 40192, "6.800e-03", "0.000326", "0.01829", "full", 1024, 50),
         ("xl-d14-4c0", 1280, 14, 3, 89600, "3.400e-03", "0.000238", "0.01641", "full", 1024, 50)]
W, DEPTH, PD = 1024, 3, 768

rows = []
for tag, d, nb, nt, S, mlr, alr, wdk, data, gb, cdp in POINT:
    f = fwd(d, nb, nt, ET, ET2)
    head = TokenFlowHead(PD, d, W, DEPTH).eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        head(torch.randn(256, PD), torch.rand(256), torch.randn(256, d))
    hf = fc.get_total_flops()
    r = (hf - 2 * 256 * d * PD) / f
    s = int(S / (1 + r)) // 32 * 32
    name = tag.replace("xl-", "dtm-")
    print(f"{tag}: forward {f / 1e9:.3f} GFLOP per image; head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M parameters, {hf / 1e9:.3f} GFLOP "
          f"per image; r {r:.5f}; steps {s} (point {S}), budget used {s * (1 + r) / S:.5f}, FLOPs {3 * f * (1 + r) * gb * s:.4g} (point {3 * f * gb * S:.4g})")
    rows.append(f"{name} {d} {nb} {nt} {s} {mlr} {alr} {5 / (float(mlr) * s):.4g} {data} {gb} {cdp}")
print("\n".join(rows))
