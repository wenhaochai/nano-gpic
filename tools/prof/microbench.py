"""Isolated re-benchmarks of the job's own ops, at the shapes the profiled trace recorded.

For every unique (op, input shapes, dtypes) of the GEMM and attention ops in a torch.profiler trace, time the op
alone (CUDA events, warm-up, median of N) and report achieved TFLOP/s against the H100 SXM dense peak. For attention
it also times every SDPA backend at that shape. Then the memory-bound tail: AdamW (foreach vs fused), the EMA update
(foreach mul+add vs foreach lerp), gradient-norm clipping, and the pageable vs pinned host-to-device batch copy.

usage: python microbench.py <trace rank0.json> <out.json>
Single GPU. Nothing here changes the job; it measures what each op could cost on its own.
"""
import collections, json, math, statistics, sys, time

import torch
import torch.nn.functional as F

PEAK_BF16 = 989.4e12   # H100 SXM dense bf16/fp16 tensor core
PEAK_TF32 = 494.7e12
PEAK_FP32 = 66.9e12
HBM_BW = 3.35e12
DT = {"c10::BFloat16": torch.bfloat16, "c10::Half": torch.float16, "float": torch.float32, "double": torch.float64,
      "bool": torch.bool, "long int": torch.int64, "int": torch.int32}
ELEM = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4, torch.bool: 1, torch.int64: 8, torch.int32: 4}


def timeit(fn, warm=5, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def load_ops(trace):
    ev = json.load(open(trace))["traceEvents"]
    ops = collections.Counter()
    gpu_ms = collections.defaultdict(float)
    ext2op = {}
    for e in ev:
        if e.get("cat") == "cpu_op":
            a = e.get("args", {})
            key = (e["name"], json.dumps(a.get("Input Dims")), json.dumps(a.get("Input type")))
            ext2op[a.get("External id")] = key
    for e in ev:
        if e.get("cat") == "kernel":
            k = ext2op.get(e.get("args", {}).get("External id"))
            if k:
                gpu_ms[k] += e.get("dur", 0) / 1e3
    for e in ev:
        if e.get("cat") == "cpu_op":
            a = e.get("args", {})
            ops[(e["name"], json.dumps(a.get("Input Dims")), json.dumps(a.get("Input type")))] += 1
    return ops, gpu_ms


def rand(shape, dtype):
    if dtype == torch.bool:
        return torch.ones(shape, dtype=torch.bool, device="cuda")
    return torch.randn(shape, dtype=dtype, device="cuda") * 0.02


def gemm_case(name, dims, types):
    """Return (callable, flops, bytes) for a GEMM op recorded in the trace, or None."""
    dts = [DT.get(t) for t in types]
    if name == "aten::mm" and len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
        (m, k), (k2, n) = dims[0], dims[1]
        a, b = rand((m, k), dts[0]), rand((k2, n), dts[1])
        return (lambda: torch.mm(a, b)), 2 * m * k * n, (m * k + k * n + m * n) * ELEM[dts[0]]
    if name == "aten::addmm" and len(dims) >= 3 and len(dims[1]) == 2:
        bias, (m, k), (k2, n) = dims[0], dims[1], dims[2]
        c, a, b = rand(tuple(bias), dts[0]), rand((m, k), dts[1]), rand((k2, n), dts[2])
        return (lambda: torch.addmm(c, a, b)), 2 * m * k * n, (m * k + k * n + m * n) * ELEM[dts[1]]
    if name == "aten::bmm" and len(dims) >= 2 and len(dims[0]) == 3:
        (bb, m, k), (_, k2, n) = dims[0], dims[1]
        a, b = rand((bb, m, k), dts[0]), rand((bb, k2, n), dts[1])
        return (lambda: torch.bmm(a, b)), 2 * bb * m * k * n, bb * (m * k + k * n + m * n) * ELEM[dts[0]]
    return None


SDPA_FWD = {"aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_cudnn_attention", "aten::scaled_dot_product_attention"}


def sdpa_cases(name, dims, types):
    """Time every SDPA backend, forward and forward+backward, at the recorded q/k/v shapes."""
    if name not in SDPA_FWD or len(dims) < 3 or len(dims[0]) != 4:
        return None
    (b, h, n, d), (_, hk, s, _) = dims[0], dims[1]
    dt = DT.get(types[0]) or torch.bfloat16
    has_mask = len(dims) > 3 and dims[3] not in ([], None)
    q = rand((b, h, n, d), dt).requires_grad_(); k = rand((b, hk, s, d), dt).requires_grad_()
    v = rand((b, hk, s, d), dt).requires_grad_()
    mask = None
    if has_mask:
        mask = torch.ones(tuple(dims[3]), dtype=torch.bool, device="cuda")
    flops_fwd = 4 * b * h * n * s * d
    res = {"shape": [b, h, n, s, d], "mask": has_mask, "gqa": hk != h, "flops_fwd": flops_fwd}
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for be in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
        try:
            with sdpa_kernel(getattr(SDPBackend, be)):
                f = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=hk != h)
                ms_f = timeit(f, 3, 10)
                g = torch.randn_like(q)
                def fb():
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=hk != h)
                    o.backward(g)
                ms_fb = timeit(fb, 3, 10)
            res[be] = {"fwd_ms": ms_f, "fwd_tflops": flops_fwd / ms_f / 1e9,
                       "fwdbwd_ms": ms_fb, "fwdbwd_tflops": 3.5 * flops_fwd / ms_fb / 1e9}
        except Exception as e:
            res[be] = {"error": type(e).__name__ + ": " + str(e)[:160]}
    return res


def tail_bench():
    """Optimizer, EMA, clipping and host copies on the denoiser's real parameter shapes."""
    sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/perf/baselines/PixelGen")
    from src.models.transformer.JiT_T2I import JiT_T2I
    den = JiT_T2I(input_size=256, in_channels=3, num_groups=24, hidden_size=1536, num_blocks=16, num_text_blocks=4,
                  patch_size=16, txt_embed_dim=2048, txt_max_length=300, bottleneck_dim=256).cuda()
    params = [p for p in den.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    for p in params:
        p.grad = torch.randn_like(p) * 1e-3
    out = {"n_params": n}
    for fused in (False, True):
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0, fused=fused, foreach=not fused)
        ms = timeit(opt.step, 3, 10)
        # AdamW reads p, g, m, v and writes p, m, v (fp32): 7 x 4 bytes per parameter
        out[f"adamw_{'fused' if fused else 'foreach'}"] = {"ms": ms, "GBps": 28 * n / ms / 1e6, "min_bytes_GB": 28 * n / 1e9}
    ema = [p.detach().clone().float() for p in params]
    cur = [p.detach() for p in params]
    def ema_mul_add():
        torch._foreach_mul_(ema, 0.9999); torch._foreach_add_(ema, cur, alpha=1e-4)
    def ema_lerp():
        torch._foreach_lerp_(ema, cur, 1e-4)
    for nm, fn in (("ema_foreach_mul_add", ema_mul_add), ("ema_foreach_lerp", ema_lerp)):
        ms = timeit(fn, 3, 10)
        out[nm] = {"ms": ms, "GBps": 12 * n / ms / 1e6, "min_bytes_GB": 12 * n / 1e9}  # read e, read p, write e
    ms = timeit(lambda: torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True), 3, 10)
    out["clip_grad_norm_foreach"] = {"ms": ms, "GBps": 12 * n / ms / 1e6}  # read g (norm), read+write g (scale)
    x = torch.randn(32, 3, 256, 256)
    xp = x.pin_memory()
    out["h2d_pageable_25MB"] = {"ms": timeit(lambda: x.cuda(non_blocking=False), 2, 10)}
    out["h2d_pinned_25MB"] = {"ms": timeit(lambda: xp.cuda(non_blocking=True), 2, 10)}
    return out


def main(trace, out_path):
    ops, gpu_ms = load_ops(trace)
    res = {"gemm": [], "sdpa": [], "tail": None, "peaks": {"bf16": PEAK_BF16, "hbm": HBM_BW}}
    for (name, dims_s, types_s), cnt in sorted(ops.items(), key=lambda kv: -gpu_ms.get(kv[0], 0)):
        dims, types = json.loads(dims_s) or [], json.loads(types_s) or []
        try:
            g = gemm_case(name, dims, types)
            if g:
                fn, flops, nbytes = g
                ms = timeit(fn)
                res["gemm"].append({"op": name, "dims": dims, "types": types, "calls_in_trace": cnt,
                                    "in_trace_gpu_ms_total": gpu_ms.get((name, dims_s, types_s), 0.0),
                                    "iso_ms": ms, "iso_tflops": flops / ms / 1e9, "flops": flops, "bytes": nbytes,
                                    "iso_eff_vs_bf16_peak": flops / ms / 1e9 / (PEAK_BF16 / 1e12)})
                continue
            s = sdpa_cases(name, dims, types)
            if s:
                s.update({"op": name, "calls_in_trace": cnt, "in_trace_gpu_ms_total": gpu_ms.get((name, dims_s, types_s), 0.0)})
                res["sdpa"].append(s)
        except Exception as e:
            res.setdefault("errors", []).append([name, dims, repr(e)[:200]])
    torch.cuda.empty_cache()
    res["tail"] = tail_bench()
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print(f"gemm cases {len(res['gemm'])}, sdpa cases {len(res['sdpa'])}, errors {len(res.get('errors', []))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
