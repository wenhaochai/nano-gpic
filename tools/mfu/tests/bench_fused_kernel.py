#!/usr/bin/env python3
"""MFU lever H: launch configurations of the fused clip + AdamW + EMA kernel (src/utils/fused_adamw_ema.py) at a width's
real parameter shapes, on one GPU. For each BLOCK:NUM_WARPS:ALIGNED: bit for bit against torch over 4 steps at those
shapes (so the 16-byte path is checked too), then the median time of torch (clip multiply + fused AdamW + lerp) and of
the kernel with and without the EMA (norm and clip coefficient included in all three).
usage: python bench_fused_kernel.py --d 1024 --configs 4096:8:1,2048:4:1
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_fused_adamw_ema as T   # noqa: E402  (its sys.path setup finds src/)
import src.utils.fused_adamw_ema as F   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--configs", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    shapes = T.model_shapes(a.d)
    n = sum(torch.Size(s).numel() for s in shapes)
    print(f"d{a.d}: {len(shapes)} tensors, {n / 1e6:.1f} M parameters")
    ref = T.run(shapes, 4, 0.0, dev, fused=False)
    t_torch = T.bench(shapes, dev, "torch")
    print(f"  torch: {t_torch:.3f} ms")
    for c in a.configs.split(","):
        b, w, al = c.split(":")
        F.BLOCK, F.NUM_WARPS, F.USE_ALIGNED = int(b), int(w), al == "1"
        mm = T.mismatches(T.run(shapes, 4, 0.0, dev, fused=True), ref)
        t1, t2 = T.bench(shapes, dev, "fused"), T.bench(shapes, dev, "fused+ema")
        print(f"  BLOCK {b:>5s} warps {w:>2s} aligned {al}: mismatches {mm}; fused {t1:.3f} ms, fused+ema {t2:.3f} ms "
              f"(torch {t_torch:.3f})", flush=True)


if __name__ == "__main__":
    main()
