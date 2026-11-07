#!/usr/bin/env python3
"""Attention backends at the ladder's shapes, with the key-padding mask (docs/mfu.md lever A).

Image blocks: q [B=32, 256, H, 64], keys = 256 image tokens + T caption tokens of which the first n_b are valid; text
blocks: q = k = v = T caption tokens. For each width's H and T in (24, 56, 168): forward and forward+backward time
(median of 30, CUDA events) and the error against an fp32 math reference with the same mask: max |out|, max |dq|,
|dk|, |dv| differences, and the largest gradient on a padded key (must be 0).
Backends: sdpa (what production runs: SDPA with a boolean mask, which picks the memory-efficient kernel), sdpa_cudnn
(SDPA, cuDNN backend forced), fa3 (src/models/layers/fa3_op.py), flex (FlexAttention, compiled, padding block mask),
and fa3_compiled (fa3 inside torch.compile with a dynamic T: the custom op's fake kernels).
usage: python attn_bench.py [--backends sdpa,sdpa_cudnn,fa3,flex,fa3_compiled] [--out x.json]
"""
import argparse, json, os, random, statistics, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen"))


def timeit(fn, warm=5, iters=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="sdpa,sdpa_cudnn,fa3,flex,fa3_compiled")
    ap.add_argument("--heads", default="10,16,20,28")
    ap.add_argument("--Ts", default="24,56,168")
    ap.add_argument("--out")
    a = ap.parse_args()
    print(torch.__version__, torch.cuda.get_device_name(0), "cudnn", torch.backends.cudnn.version(), flush=True)
    dev = "cuda"
    B, D, N = 32, 64, 256
    backends = a.backends.split(",")
    fa3 = None
    if any(b.startswith("fa3") for b in backends):
        from src.models.layers.fa3_op import fa3_attention
        fa3 = fa3_attention
    flex = None
    if "flex" in backends:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        flex_c = torch.compile(flex_attention, dynamic=False)
    fa3_c = torch.compile(lambda q, k, v, s: fa3(q, k, v, s), dynamic=True) if "fa3_compiled" in backends else None
    rows = []
    rng = random.Random(0)
    for kind in ("image", "text"):
        for H in [int(h) for h in a.heads.split(",")]:
            for T in [int(t) for t in a.Ts.split(",")]:
                Nq = N if kind == "image" else T
                Nk = N + T if kind == "image" else T
                top = rng.randint(T - 7, T)
                lens = [top] + [rng.randint(max(1, T - 16), top) for _ in range(B - 1)]
                n_valid = torch.tensor([(N if kind == "image" else 0) + l for l in lens], device=dev, dtype=torch.int32)
                keymask = torch.arange(Nk, device=dev)[None] < n_valid[:, None]             # [B, Nk]
                g = torch.Generator(device=dev).manual_seed(H * 1000 + T)
                q = torch.randn(B, Nq, H, D, device=dev, generator=g, dtype=torch.bfloat16)
                k = torch.randn(B, Nk, H, D, device=dev, generator=g, dtype=torch.bfloat16)
                v = torch.randn(B, Nk, H, D, device=dev, generator=g, dtype=torch.bfloat16)
                do = torch.randn(B, Nq, H, D, device=dev, generator=g, dtype=torch.bfloat16)
                # fp32 reference
                qr, kr, vr = (t.float().transpose(1, 2).requires_grad_() for t in (q, k, v))
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    ref = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=keymask[:, None, None, :])
                ref.backward(do.float().transpose(1, 2))
                ref_o, ref_g = ref.detach().transpose(1, 2), [t.grad.transpose(1, 2) for t in (qr, kr, vr)]
                valid_flops = sum(4 * Nq * int(n) * D * H for n in n_valid.tolist())
                for be in backends:
                    if be == "sdpa":
                        def f(q, k, v):
                            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                                  attn_mask=keymask[:, None, None, :]).transpose(1, 2)
                    elif be == "sdpa_cudnn":
                        def f(q, k, v):
                            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                                return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                                      attn_mask=keymask[:, None, None, :]).transpose(1, 2)
                    elif be == "fa3":
                        def f(q, k, v):
                            return fa3(q, k, v, n_valid)
                    elif be == "fa3_compiled":
                        def f(q, k, v):
                            return fa3_c(q, k, v, n_valid)
                    elif be == "flex":
                        nv = n_valid

                        def mask_mod(b, h, qi, ki):
                            return ki < nv[b]
                        bm = create_block_mask(mask_mod, B, None, Nq, Nk, device=dev)

                        def f(q, k, v):
                            return flex_c(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=bm).transpose(1, 2)
                    row = {"kind": kind, "H": H, "T": T, "Nq": Nq, "Nk": Nk, "backend": be}
                    try:
                        qq, kk, vv = (t.detach().clone().requires_grad_() for t in (q, k, v))
                        o = f(qq, kk, vv)
                        o.backward(do)
                        torch.cuda.synchronize()
                        row["err_out"] = (o.float() - ref_o).abs().max().item()
                        row["ref_out"] = ref_o.abs().max().item()
                        for name, gg, rg in zip("qkv", (qq.grad, kk.grad, vv.grad), ref_g):
                            row[f"err_d{name}"] = (gg.float() - rg).abs().max().item()
                        pad = ~keymask
                        row["pad_grad"] = max(kk.grad[pad].abs().max().item() if pad.any() else 0.0,
                                              vv.grad[pad].abs().max().item() if pad.any() else 0.0)
                        qd, kd, vd = q.detach(), k.detach(), v.detach()
                        row["fwd_ms"] = timeit(lambda: f(qd, kd, vd))
                        qg, kg, vg = (t.detach().clone().requires_grad_() for t in (q, k, v))

                        def fb():
                            f(qg, kg, vg).backward(do)
                        row["fwdbwd_ms"] = timeit(fb)
                        row["fwd_tflops"] = valid_flops / row["fwd_ms"] / 1e9
                        row["bwd_tflops"] = 2.5 * valid_flops / (row["fwdbwd_ms"] - row["fwd_ms"]) / 1e9
                    except Exception as e:
                        row["error"] = repr(e)[:300]
                    rows.append(row)
                    if "error" in row:
                        print(f"{kind:5s} H={H:2d} T={T:3d} {be:13s} ERROR {row['error']}", flush=True)
                    else:
                        print(f"{kind:5s} H={H:2d} T={T:3d} {be:13s} fwd {row['fwd_ms']:7.3f} ms ({row['fwd_tflops']:5.0f} TF/s) "
                              f"fwd+bwd {row['fwdbwd_ms']:7.3f} ms (bwd {row['bwd_tflops']:5.0f} TF/s)  err out "
                              f"{row['err_out']:.2e} dq {row['err_dq']:.2e} dk {row['err_dk']:.2e} dv {row['err_dv']:.2e} "
                              f"pad_grad {row['pad_grad']:.1e}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
