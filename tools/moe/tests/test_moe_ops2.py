#!/usr/bin/env python3
"""GPU test of the v2 MoE operators (src/models/layers/moe_ops.py v2, GPIC_MOE_KERNELS=2: sums in token order through the inverse slot map)
against the torch path of the layer (fp32 atomics). The ops alone: the inverse map against the slot table, the token sum against index_add in
fp64, the gather's backward. The layer, every route (and 64 experts with 8 active), fp32 (no autocast), bf16 autocast and bf16 under
torch.compile: output and every gradient agree with the torch path (fp32 within 1e-5 of the largest value: the order of a token's fp32
terms; bf16 within 2e-2), two calls of the v2 path are bit for bit equal, and the compiled v2 path has no graph break.
usage: python test_moe_ops2.py   (a CUDA GPU; a vis-node A40 is enough)"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
os.environ["GPIC_MOE_KERNELS"] = "0"
import src.models.layers.moe as moe_mod   # noqa: E402
from src.models.layers import moe_ops     # noqa: E402,F401
import torch._dynamo as dynamo           # noqa: E402
dev = "cuda"
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


rel = lambda a, b: float((a.double() - b.double()).abs().max()) / max(float(b.double().abs().max()), 1e-12)

# ---- the ops alone ----
torch.manual_seed(0)
M, n, cap, D = 300, 6, 70, 96
S = n * cap
idx = torch.stack([torch.randperm(M, device=dev)[:cap] for _ in range(n)])   # each expert takes cap distinct tokens
valid = torch.rand(n, cap, device=dev) > 0.2
flat = idx.reshape(-1); vf = valid.reshape(-1)
inv = torch.ops.gpic.moe_inv(flat, vf, M, n)
ref_inv = torch.full((M, n), -1, dtype=torch.int32, device=dev)
for e in range(n):
    for r in range(cap):
        if valid[e, r]:
            ref_inv[idx[e, r], e] = r
check("inv = the slot table's inverse (valid slots only)", torch.equal(inv, ref_inv))
o = torch.randn(S, D, device=dev, dtype=torch.bfloat16)
w = torch.rand(S, device=dev) * vf
ref = torch.zeros(M, D, dtype=torch.float64, device=dev).index_add_(0, flat, o.double() * w.double()[:, None])
got = torch.ops.gpic.moe_combine2(o, w, flat, inv, M, True)
check("combine2 = fp64 index_add, then bf16", rel(got.float(), ref.float()) < 8e-3 and got.dtype == torch.bfloat16, f"{rel(got.float(), ref.float()):.1e}")
ge = torch.randn(S, D, device=dev) * vf[:, None]          # an empty slot's gradient row is 0 in the layer
x = torch.randn(M, D, device=dev).requires_grad_(True); x2 = x.detach().clone().requires_grad_(True)
x.index_select(0, flat).backward(ge); torch.ops.gpic.moe_gather2(x2, flat, inv).backward(ge)
check("gather2 backward = index_add (fp32)", rel(x2.grad, x.grad) < 1e-6, f"{rel(x2.grad, x.grad):.1e}")

# ---- the layer, v2 against torch ----
CASES = (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("eci", 16, 2, {}), ("noise", 8, 1, {"t_edges": [i / 8 for i in range(1, 8)], "capacity": 2.0}),
         ("pos", 8, 1, {}), ("hash", 8, 1, {}), ("ecb", 64, 8, {}))
d, T, B = 128, 256, 8
for route, n, G, kw in CASES:
    for mode in ("fp32", "bf16", "bf16-compiled"):
        torch.manual_seed(0)
        m = moe_mod.MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T, **kw).to(dev).train()
        if m.router is not None:
            torch.nn.init.normal_(m.router.weight, std=0.05)
        xin = torch.randn(B, T, d, device=dev); tt = torch.rand(B, device=dev); gy = torch.randn(B, T, d, device=dev)

        def run(kernels, mod):
            moe_mod._KERNELS = kernels
            xx = xin.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32"):
                y = mod(xx, tt)
            y.float().backward(gy)
            gs = [xx.grad] + [p.grad.clone() for p in m.parameters() if p.grad is not None]
            for p in m.parameters():
                p.grad = None
            return y.detach().float(), gs
        mod = m
        if mode == "bf16-compiled":
            dynamo.reset(); mod = torch.compile(m)
        yt, gt = run(0, mod)
        if mode == "bf16-compiled":
            dynamo.reset()
            moe_mod._KERNELS = 2
            breaks = dynamo.explain(m)(xin.clone().requires_grad_(True), tt).graph_break_count
            dynamo.reset(); mod = torch.compile(m)
        else:
            breaks = 0
        yk, gk = run(2, mod); yk2, gk2 = run(2, mod)
        tol = 1e-5 if mode == "fp32" else 2e-2
        e_y, e_g = rel(yk, yt), max(rel(a, b) for a, b in zip(gk, gt))
        same = torch.equal(yk, yk2) and all(torch.equal(a, b) for a, b in zip(gk, gk2))
        check(f"layer {route} n{n} G{G} {mode}: v2 = torch path", e_y < tol and e_g < tol and same and breaks == 0,
              f"output {e_y:.1e}, gradients {e_g:.1e} (tolerance {tol:.0e}); repeat bit for bit {same}; graph breaks {breaks}")
# ---- evaluation (routes tc and ecb route by threshold there: a slot table as wide as the busiest expert, eager) ----
for route, n, G, kw in (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("ecb", 64, 8, {})):
    for mode in ("fp32", "bf16"):
        torch.manual_seed(0)
        m = moe_mod.MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T, **kw).to(dev).eval()
        torch.nn.init.normal_(m.router.weight, std=0.05)
        if route == "ecb":   # measured thresholds, as a trained layer has (the boundary score of each expert's top tokens)
            m.moe_thr.copy_(torch.rand(n, device=dev) * 0.2 + 0.45); m.moe_thr_set.fill_(True)
        xin = torch.randn(B, T, d, device=dev); tt = torch.rand(B, device=dev)
        outs = []
        for k in (0, 2, 2):
            moe_mod._KERNELS = k
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32"):
                outs.append(m(xin, tt).float())
        tol = 1e-5 if mode == "fp32" else 2e-2
        check(f"evaluation {route} n{n} G{G} {mode}: v2 = torch path", rel(outs[1], outs[0]) < tol and torch.equal(outs[1], outs[2]),
              f"{rel(outs[1], outs[0]):.1e}; repeat bit for bit {torch.equal(outs[1], outs[2])}")
moe_mod._KERNELS = 0
print("MOE_OPS2_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
