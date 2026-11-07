"""Which EMA update is closer to exact arithmetic: base (foreach mul then add) or C3 (foreach lerp)?

Both compute ema <- decay * ema + (1 - decay) * p in fp32. The correctness check flagged C3's per-tensor EMA sums
(maxabs 1.56e-2 against a tolerance of 3.3e-4 that came from runs sharing the base formula). Here both formulas run on
the GPU from the same start for K steps of AdamW-sized parameter moves (lr per element), next to an fp64 reference of
the same recurrence. The error of each formula against the reference decides which rounding is better.

python tools/test_ema_precision.py [--steps 10] [--decay 0.9999] [--lr 1e-4]
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    # shapes of the largest denoiser weights (MLP 1536x6144 both ways, fused qkv 4608x1536), a norm weight, a bias
    cases = [((6144, 1536), 0.02), ((1536, 6144), 0.02), ((4608, 1536), 0.02), ((1536,), 1.0), ((1536,), 0.0)]
    print(f"steps {a.steps} decay {a.decay} lr {a.lr}")
    print(f"{'shape':>14s} {'init':>5s} {'rule':>5s} {'max|err|':>10s} {'mean err':>10s} {'sum err (fp64)':>15s} "
          f"{'fp32-sum diff':>14s}")
    for shape, std in cases:
        p = torch.randn(shape, device=dev) * std if std else torch.zeros(shape, device=dev)
        base, lerp, ref = p.clone(), p.clone(), p.double()
        for _ in range(a.steps):
            p = p - a.lr * torch.sign(torch.randn_like(p))
            torch._foreach_mul_([base], a.decay)
            torch._foreach_add_([base], [p], alpha=1.0 - a.decay)
            torch._foreach_lerp_([lerp], [p], 1.0 - a.decay)
            ref = a.decay * ref + (1.0 - a.decay) * p.double()
        ref_sum32 = ref.float().sum().item()
        for name, e in (("base", base), ("lerp", lerp)):
            err = e.double() - ref
            print(f"{str(tuple(shape)):>14s} {std:5.2f} {name:>5s} {err.abs().max().item():10.3e} {err.mean().item():10.3e} "
                  f"{err.sum().item():15.3e} {e.sum().item() - ref_sum32:14.3e}")
        d = (base.sum() - lerp.sum()).item()
        print(f"{'':>14s} {'':>5s} {'b-l':>5s} fp32 per-tensor sum difference {d:.3e} (what the correctness check compares)")


if __name__ == "__main__":
    main()
