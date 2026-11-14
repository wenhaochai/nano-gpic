#!/usr/bin/env python3
"""One MoE layer, forward + backward, under several operator settings (2026-11-14: v4 and the router's score operator), on one GPU.

As bench_moe_layer.py --mp bf16 (FSDP2's mixed precision: the layer's parameters and input in bf16, the router fp32, autocast on,
torch.compile in the default mode, M = images x 256 tokens), but only the whole layer, for each arm of --arms:
  <kernels>[+wt][+rt]   GPIC_MOE_KERNELS, GPIC_MOE_BMM_WT=1 (+wt), GPIC_MOE_ROUTER=1 (+rt); e.g. 3, 3+wt, 3+wt+rt, 4+wt, 4+wt+rt
Per arm: the median time of --iters calls (CUDA events) after 10 warm-up calls, the ratio to the dense SwiGLU of hidden 4d, the peak
allocated memory of one call above the resting level, and with --profile the top kernels of 5 calls. One JSON line to --out.
usage: bench_moe_arms.py --d 1024 --images 128 --experts 16 --active 2 --route ecb [--arms 3,3+wt,3+wt+rt,4+wt,4+wt+rt] [--profile]"""
import argparse, json, os, statistics, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
import src.models.layers.moe as moe_mod                    # noqa: E402
from src.models.layers.moe import MoEFeedForward          # noqa: E402
from src.models.layers.swiglu import SwiGLU               # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=1024); p.add_argument("--images", type=int, default=128)
p.add_argument("--experts", type=int, default=16); p.add_argument("--active", type=int, default=2)
p.add_argument("--route", default="ecb"); p.add_argument("--capacity", type=float, default=1.5)
p.add_argument("--arms", default="3,3+wt,3+wt+rt,4+wt,4+wt+rt")
p.add_argument("--out", default=""); p.add_argument("--profile", action="store_true"); p.add_argument("--iters", type=int, default=30)
a = p.parse_args()
dev = "cuda"
torch.manual_seed(0)
d, H, T = a.d, 4 * a.d, 256
M = a.images * T
x0 = torch.randn(a.images, T, d, device=dev, dtype=torch.bfloat16)
tt = torch.rand(a.images, device=dev)


def fwd_bwd(mod, extra=()):
    x = x0.clone().requires_grad_(True)
    g = torch.randn_like(x0, dtype=torch.bfloat16)
    params = list(mod.parameters())

    def run():
        for q in params:   # as an optimizer step leaves them
            q.grad = None
        x.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x, *extra)
        y.backward(g)
    return run


def timed(fn, iters, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), peak


res = dict(d=d, images=a.images, tokens=M, experts=a.experts, active=a.active, route=a.route, gpu=torch.cuda.get_device_name(), arms={})
dense = SwiGLU(d, H).to(dev).to(torch.bfloat16)
res["dense_ms"], res["dense_peak_gb"] = timed(fwd_bwd(torch.compile(dense)), a.iters)
kw = dict(experts=a.experts, active=a.active, route=a.route, capacity=a.capacity, tokens=T)
if a.route == "noise":
    kw["t_edges"] = [(i + 0.5) / a.experts for i in range(a.experts - 1)]
moe = MoEFeedForward(d, H, **kw).to(dev).train()
if moe.router is not None:
    torch.nn.init.normal_(moe.router.weight, std=0.02)
for name, q in moe.named_parameters():   # FSDP2's param_dtype: the sharded parameters in bf16; the router (fsdp_keep) stays fp32
    if not name.startswith("router."):
        q.data = q.data.to(torch.bfloat16)
print(f"{res['gpu']}: d {d}, {M} tokens, {a.route} {a.experts} experts, {a.active} active (expert hidden {moe.h}); dense {res['dense_ms']:.3f} ms", flush=True)
for arm in a.arms.split(","):
    f = arm.split("+")
    moe_mod._KERNELS, moe_mod._BMM_WT, moe_mod._ROUTER = int(f[0]), "wt" in f[1:], "rt" in f[1:]
    torch._dynamo.reset()
    run = fwd_bwd(torch.compile(moe), (tt,))
    ms, peak = timed(run, a.iters)
    res["arms"][arm] = dict(ms=ms, ratio=ms / res["dense_ms"], peak_gb=peak)
    line = f"  {arm:10s} {ms:8.3f} ms  x{ms / res['dense_ms']:.3f} dense  peak +{peak:.2f} GiB"
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                run()
            torch.cuda.synchronize()
        ka = sorted(prof.key_averages(), key=lambda k: -k.device_time_total)[:12]
        res["arms"][arm]["top"] = [(k.key[:70], round(k.device_time_total / 5 / 1000, 3), k.count // 5) for k in ka]
        line += "\n" + "\n".join(f"      {ms_:7.3f} ms x{c:<3d} {name}" for name, ms_, c in res["arms"][arm]["top"])
    print(line, flush=True)
    moe.moe_stat.zero_()
moe_mod._KERNELS, moe_mod._BMM_WT, moe_mod._ROUTER = 0, False, False
if a.out:
    with open(a.out, "a") as fo:
        fo.write(json.dumps(res) + "\n")
