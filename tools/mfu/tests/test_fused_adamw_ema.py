#!/usr/bin/env python3
"""MFU lever H (src/utils/fused_adamw_ema.py), on one GPU: the fused kernel against torch's clip_grad_norm_ (foreach) +
AdamW(fused=True) + _foreach_lerp_ (SimpleEMA), bit for bit, then the time of both at a width's real parameter shapes.

1. Every FMA placement (FMA_M, FMA_V, FMA_E) over 6 steps on tensors of awkward sizes (1, 3, 4095, 4097, ...), gradients at
   three scales, a max_norm that clips on some steps and not on others, weight decay 0 and 0.1: mismatching elements of
   param, exp_avg, exp_avg_sq and EMA per variant. PASS needs one variant with no mismatch at weight decay 0 (production).
2. Time at the parameter shapes of JiT_T2I at --d (CUDA events, median of 30): torch (clip multiply + fused AdamW + lerp)
   against the kernel with and without the EMA.
usage: python test_fused_adamw_ema.py [--d 1024]
"""
import argparse, itertools, os, statistics, sys

import torch

sys.path.insert(0, os.environ.get("PIXGEN_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
import src.utils.fused_adamw_ema as F

SIZES = {640: (9, 2), 768: (10, 2), 896: (11, 2), 1024: (12, 3), 1280: (14, 3), 1536: (16, 4), 1792: (18, 4)}


def make(shapes, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    ps = [torch.randn(s, device=dev, generator=g) * 0.02 for s in shapes]
    return ps


def run(shapes, steps, wd, dev, fused, max_norm=1.0, decay=0.9999):
    """One trajectory from seed 0: returns the final params, exp_avg, exp_avg_sq, EMA."""
    ps = [torch.nn.Parameter(p) for p in make(shapes, dev, 0)]
    ema = [p.detach().clone() for p in ps]
    opt = torch.optim.AdamW(ps, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd, fused=True)
    if fused:
        opt = F.FusedAdamWEMA.like(opt, [{"params": ps}])
        opt.ema_of, opt.ema_weight = dict(zip(ps, ema)), 1.0 - decay
    g = torch.Generator(device=dev).manual_seed(1)
    for k in range(steps):
        scale = (1e-3, 0.05, 3.0)[k % 3]   # grad norms below and above max_norm
        for p in ps:
            p.grad = torch.randn(p.shape, device=dev, generator=g) * scale
        if fused:
            grads = [p.grad for p in ps if p.grad is not None]
            total_norm = torch.nn.utils.clip_grad._get_total_norm(grads, 2.0, False, True)
            clip_coef = max_norm / (total_norm + 1e-6)
            opt.clip_coef = torch.clamp(clip_coef, max=1.0)
            opt.step()
        else:
            torch.nn.utils.clip_grad_norm_(ps, max_norm, foreach=True)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema, ps, 1.0 - decay)
    st = [opt.state[p] for p in ps]
    return ([p.detach() for p in ps], [s["exp_avg"] for s in st], [s["exp_avg_sq"] for s in st], ema)


def mismatches(a, b):
    out = []
    for xa, xb in zip(a, b):
        out.append(sum(int((x.view(torch.int32) != y.view(torch.int32)).sum()) for x, y in zip(xa, xb)))
    return out


def model_shapes(d):
    from src.models.transformer.JiT_T2I import JiT_T2I
    nb, nt = SIZES[d]
    with torch.device("meta"):
        net = JiT_T2I(input_size=256, in_channels=3, num_groups=d // 64, hidden_size=d, num_blocks=nb, num_text_blocks=nt,
                      patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
    return [tuple(p.shape) for p in net.parameters() if p.requires_grad]


def bench(shapes, dev, mode, iters=30):
    ps = [torch.nn.Parameter(p) for p in make(shapes, dev, 0)]
    for p in ps:
        p.grad = torch.randn_like(p) * 1e-3
    ema = [p.detach().clone() for p in ps]
    opt = torch.optim.AdamW(ps, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    if mode != "torch":
        opt = F.FusedAdamWEMA.like(opt, [{"params": ps}])
        if mode == "fused+ema":
            opt.ema_of, opt.ema_weight = dict(zip(ps, ema)), 1e-4
    grads = [p.grad for p in ps]
    ts = []
    for k in range(iters + 5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        total_norm = torch.nn.utils.clip_grad._get_total_norm(grads, 2.0, False, True)
        coef = torch.clamp(1.0 / (total_norm + 1e-6), max=1.0)
        if mode == "torch":
            torch._foreach_mul_(grads, coef)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema, ps, 1e-4)
        else:
            opt.clip_coef = coef
            opt.step()
            if mode == "fused":
                with torch.no_grad():
                    torch._foreach_lerp_(ema, ps, 1e-4)
        e1.record()
        torch.cuda.synchronize()
        if k >= 5:
            ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    a = ap.parse_args()
    dev = torch.device("cuda")
    shapes = [(1,), (3,), (4095,), (4096,), (4097,), (7, 129), (1024, 1024), (3, 1000, 7), (8191,), (65536 + 3,)]
    ok_variant = None
    for wd in (0.0, 0.1):
        ref = run(shapes, 6, wd, dev, fused=False)
        print(f"weight decay {wd}: mismatching elements of (param, exp_avg, exp_avg_sq, ema) of "
              f"{sum(torch.Size(s).numel() for s in shapes)} each")
        for fm, fv, fe in itertools.product((0, 1, 2), (0, 1, 2), (0, 1)):
            F.FMA_M, F.FMA_V, F.FMA_E = fm, fv, fe
            mm = mismatches(run(shapes, 6, wd, dev, fused=True), ref)
            print(f"  FMA_M={fm} FMA_V={fv} FMA_E={fe}: {mm}")
            if wd == 0.0 and sum(mm) == 0 and ok_variant is None:
                ok_variant = (fm, fv, fe)
    print(f"bitwise variant at weight decay 0: {ok_variant}")
    if ok_variant is not None:
        F.FMA_M, F.FMA_V, F.FMA_E = ok_variant
    ms = model_shapes(a.d)
    n = sum(torch.Size(s).numel() for s in ms)
    print(f"d{a.d}: {len(ms)} tensors, {n / 1e6:.1f} M parameters")
    for mode in ("torch", "fused", "fused+ema"):
        t = bench(ms, dev, mode)
        print(f"  {mode:10s} {t:.3f} ms per step (norm + clip + AdamW + EMA lerp)")
    print("FUSED_OPT_TEST " + ("PASS" if ok_variant is not None else "FAIL"))
    sys.exit(0 if ok_variant is not None else 1)


if __name__ == "__main__":
    main()
