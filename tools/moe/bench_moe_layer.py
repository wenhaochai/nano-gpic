#!/usr/bin/env python3
"""Where an MoE layer's time goes (owner 2026-11-13: 「Infra上也要努力的 moe算子」), on one GPU, at the ladder's shapes:
the image blocks' feed-forward at M = images x 256 tokens, bf16 autocast, torch.compile as in training (default mode), forward + backward.
Per configuration it times (CUDA events, median of 30 after 10 warm-up calls):
  dense    the SwiGLU of hidden 4d (the reference: the same active FLOPs)
  moe      MoEFeedForward (src/models/layers/moe.py) with the given route, experts and active
  parts    the MoE layer's parts, each compiled on its own: route (router GEMM + selection + slot table), gather (the experts' input rows),
           experts (the two bmm and SwiGLU), combine (gate and scatter-add), and their backward where they have one
  muon     one Muon step (Newton-Schulz, bf16, 5 steps) of this layer's expert matrices on this GPU, against the dense layer's
and prints the active TFLOPS (6 x d x H x tokens per forward + backward pass x 3), the ratio to dense, and the top kernels of one profiled
MoE call (torch.profiler, CUDA time). One JSON line per configuration goes to --out.
usage: bench_moe_layer.py --d 1024 --images 128 --experts 16 --active 2 --route ecb [--out file.jsonl] [--profile]"""
import argparse, json, math, os, statistics, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
from src.models.layers.moe import MoEFeedForward          # noqa: E402
from src.models.layers.swiglu import SwiGLU               # noqa: E402
from src.optim.muon import msign                          # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=1024); p.add_argument("--images", type=int, default=128)
p.add_argument("--experts", type=int, default=16); p.add_argument("--active", type=int, default=2)
p.add_argument("--route", default="ecb"); p.add_argument("--capacity", type=float, default=1.5)
p.add_argument("--out", default=""); p.add_argument("--profile", action="store_true")
p.add_argument("--iters", type=int, default=30)
a = p.parse_args()
dev = "cuda"
torch.manual_seed(0)
d, H, T = a.d, 4 * a.d, 256
M = a.images * T
x0 = torch.randn(a.images, T, d, device=dev)
tt = torch.rand(a.images, device=dev)


def timed(fn, iters=a.iters, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def fwd_bwd(mod, extra=()):
    x = x0.clone().requires_grad_(True)
    g = torch.randn_like(x0, dtype=torch.bfloat16)
    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(x, *extra)
        y.backward(g)
    return run


res = dict(d=d, H=H, images=a.images, tokens=M, experts=a.experts, active=a.active, route=a.route, gpu=torch.cuda.get_device_name(),
           torch=torch.__version__)
flops = 3 * 6 * d * H * M   # SwiGLU forward: 4 d H (w12, 2H x d) + 2 d H (w3) = 6 d H per token; x3 for forward + backward
dense = SwiGLU(d, H).to(dev)
dense_c = torch.compile(dense)
res["dense_ms"] = timed(fwd_bwd(dense_c))
kw = dict(experts=a.experts, active=a.active, route=a.route, capacity=a.capacity, tokens=T)
if a.route == "noise":
    kw["t_edges"] = [(i + 0.5) / a.experts for i in range(a.experts - 1)]
moe = MoEFeedForward(d, H, **kw).to(dev).train()
if moe.router is not None:
    torch.nn.init.normal_(moe.router.weight, std=0.02)
moe_c = torch.compile(moe)
res["moe_ms"] = timed(fwd_bwd(moe_c, (tt,)))
res["dense_tflops"] = flops / res["dense_ms"] / 1e9
res["moe_tflops_active"] = flops / res["moe_ms"] / 1e9
res["ratio"] = res["moe_ms"] / res["dense_ms"]

# ---- the parts, each compiled on its own (forward + backward where it has one) ----
xb = x0.to(torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    idx, gate = moe.slots(xb, tt)
moe.moe_stat.zero_()
n, h, cap = moe.n, moe.h, idx.shape[1]
res["cap"] = cap

def part_route():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return moe.slots(xb, tt)
route_c = torch.compile(part_route)
res["route_ms_fwd"] = timed(route_c)
moe.moe_stat.zero_()
flat = idx.reshape(-1)
xg = xb.reshape(M, d).detach().requires_grad_(True)
def part_gather():   # as MoEFeedForward._experts: an fp32 round trip, so the backward scatter-adds into an fp32 gradient
    y = xg.float().index_select(0, flat).to(xg.dtype)
    y.backward(torch.ones_like(y))
gather_c = torch.compile(part_gather)
res["gather_ms"] = timed(gather_c)
xe = xb.reshape(M, d).index_select(0, flat).view(n, cap, d).detach().requires_grad_(True)
W12 = moe.e12.detach().to(torch.bfloat16).view(n, 2 * h, d).requires_grad_(True)
W3 = moe.e3t.detach().to(torch.bfloat16).view(n, h, d).requires_grad_(True)
ge = torch.randn(n, cap, d, device=dev, dtype=torch.bfloat16)
def part_experts():
    a1, a2 = torch.bmm(xe, W12.mT).chunk(2, dim=-1)
    o = torch.bmm(F.silu(a1) * a2, W3)
    o.backward(ge)
experts_c = torch.compile(part_experts)
res["experts_ms"] = timed(experts_c)
res["experts_tflops"] = flops / res["experts_ms"] / 1e9
o = torch.randn(n, cap, d, device=dev, dtype=torch.bfloat16).requires_grad_(True)
gt = (gate if gate is not None else torch.ones(n, cap, device=dev)).detach()
gy = torch.randn(M, d, device=dev, dtype=torch.bfloat16)
def part_combine():   # as MoEFeedForward._experts: gate and sum in fp32
    y = (o.float() * gt.unsqueeze(-1).float()).reshape(n * cap, d)
    out = torch.zeros(M, d, dtype=torch.float32, device=dev).index_add_(0, flat, y).to(o.dtype)
    out.backward(gy)
combine_c = torch.compile(part_combine)
res["combine_ms"] = timed(combine_c)

# ---- Muon on this layer's matrices (one GPU does all of them; under DDP each rank does 1/8) ----
def ns_units(shapes):
    groups = {}
    for s in shapes:
        groups.setdefault(s, 0); groups[s] += 1
    mats = {s: torch.randn(c, *s, device=dev) for s, c in groups.items()}
    def run():
        for s, m in mats.items():
            msign(m, steps=5, dtype=torch.bfloat16, fused=True)
    return run
res["muon_dense_ms"] = timed(ns_units([(H, d), (H, d), (H, d)]), iters=10, warm=3)     # w12 as 2 parts [H, d], w3 [d, H] (tall once transposed)
res["muon_moe_ms"] = timed(ns_units([(h, d)] * (3 * n)), iters=10, warm=3)             # e12: 2n units [h, d]; e3t: n units [h, d]
res["params_ffn_dense"] = 3 * d * H; res["params_ffn_moe"] = 3 * n * h * d + n * d * (moe.router is not None)

if a.profile:
    from torch.profiler import profile, ProfilerActivity
    run = fwd_bwd(moe_c, (tt,))
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            run()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    tot = sum(k.device_time_total for k in ka) or 1
    top = sorted(ka, key=lambda k: -k.device_time_total)[:14]
    res["top_kernels"] = [(k.key[:90], round(k.device_time_total / 5 / 1000, 3), round(100 * k.device_time_total / tot, 1)) for k in top]

print(json.dumps(res), flush=True)
s = res
print(f"d{d} images {a.images} ({M} tokens) {a.route} n{a.experts} G{a.active} (h {h}, cap {cap}): dense {s['dense_ms']:.2f} ms "
      f"({s['dense_tflops']:.0f} TFLOPS), moe {s['moe_ms']:.2f} ms ({s['moe_tflops_active']:.0f} active TFLOPS), ratio {s['ratio']:.3f}; parts: "
      f"route {s['route_ms_fwd']:.2f}, gather {s['gather_ms']:.2f}, experts {s['experts_ms']:.2f} ({s['experts_tflops']:.0f} TFLOPS), combine "
      f"{s['combine_ms']:.2f} ms; Muon NS dense {s['muon_dense_ms']:.2f} ms, moe {s['muon_moe_ms']:.2f} ms", flush=True)
for k in s.get("top_kernels", []):
    print("   ", k, flush=True)
if a.out:
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
