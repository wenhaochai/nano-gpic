"""FLOP-matched steps of ARTM on the final ladder (docs/artm_ladder.md; as tools/dtm_ladder/steps.py for the DTM head).

ARTM's forward FLOPs per training image, with the ladder's own per-part formula (tmp/ladder_prof/all_points.py fwd_parts, mean caption length ET):
2N = 512 image tokens through the image blocks (the context X_k and the shifted target X_{k+1}); attention counted on the keys the mask allows (a context
row sees the N context keys, target row i sees the N context keys and i + 1 target keys; every row sees the caption), so the mean image keys per row are
(N + N + (N + 1) / 2) / 2; the caption's key/value projections and the text blocks as the point model's; the patch embedding on 2N tokens; no final
linear; the head (TokenFlowHead, FlopCounterMode) on the N target patches. Steps = point steps x point FLOPs / ARTM FLOPs, a multiple of 32; Muon decay =
5 / (Muon LR x steps). usage (vis node): tools/gmflow_ladder/vis.sh tools/artm_ladder/steps.py"""
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.transformer.JiT_T2I import TokenFlowHead

WC = "/scratch/gpfs/GROUP/USER"
src = open(f"{WC}/tmp/ladder_prof/all_points.py").read()
exec(src[src.index("import glob"):src.index("import yaml")])
exec(src[src.index("WC = "):src.index("def params_formula")])   # ET, ET2; L, P, B; fwd_parts(), fwd()

POINT = [("xl-d10-0p25c0", 768, 10, 2, 21248, "1.360e-02", "0.000501", "0.0173", "full", 1024, 50),
         ("xl-d12-1c0", 1024, 12, 3, 40192, "6.800e-03", "0.000326", "0.01829", "full", 1024, 50),
         ("xl-d14-4c0", 1280, 14, 3, 89600, "3.400e-03", "0.000238", "0.01641", "full", 1024, 50)]
N, PD = 256, 768


def artm_fwd(d, nb, nt, head_flops):
    Lx = 2 * N
    keys = (N + N + (N + 1) / 2) / 2
    img_lin = nb * 32 * Lx * d * d
    kvy = nb * 4 * ET * d * d
    img_attn = nb * 4 * Lx * (keys + ET) * d
    txt = nt * 32 * ET * d * d + nt * 4 * ET2 * d
    emb = 2 * Lx * PD * 256 + 2 * Lx * 256 * d + 2 * ET * 2048 * d   # bottleneck patch embedding (P -> 256 -> d) on 2N tokens; caption embedding
    return img_lin + kvy + img_attn + txt + emb + head_flops


rows = []
for tag, d, nb, nt, S, mlr, alr, wdk, data, gb, cdp in POINT:
    f = fwd(d, nb, nt, ET, ET2)
    head = TokenFlowHead(PD, d, 1024, 3).eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        head(torch.randn(N, PD), torch.rand(N), torch.randn(N, d))
    hf = fc.get_total_flops()
    a = artm_fwd(d, nb, nt, hf)
    s = int(S * f / a) // 32 * 32
    print(f"{tag}: point {f / 1e9:.2f} GFLOP per image; ARTM {a / 1e9:.2f} ({a / f:.3f} x; head {hf / 1e9:.2f}); steps {s} (point {S}); "
          f"FLOPs {3 * a * gb * s:.4g} (point {3 * f * gb * S:.4g})")
    rows.append(f"{tag.replace('xl-', 'artm-')} {d} {nb} {nt} {s} {mlr} {alr} {5 / (float(mlr) * s):.4g} {data} {gb} {cdp}")
print("\n".join(rows))
