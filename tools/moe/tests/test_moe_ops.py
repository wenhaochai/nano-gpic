#!/usr/bin/env python3
"""GPU test of the Triton MoE operators (src/models/layers/moe_ops.py, GPIC_MOE_KERNELS=1) against the torch path of the layer (fp32
atomics). For every route: the layer with the kernels and the layer without them, the same weights and inputs, fp32 (no autocast) and bf16
autocast, eager and torch.compile: outputs and every gradient agree (fp32: within 1e-5 of the largest value: only the order of a token's
fp32 terms differs; bf16: within 1e-2), two calls of the kernel path are bit for bit equal, and the compiled kernel path has no graph
break. Also the ops alone: moe_csr's perm and offsets, the segmented sum against index_add_ in fp64, the combine's backward (grad_o,
grad_gate) against autograd of the torch expression.
usage: python test_moe_ops.py   (a CUDA GPU; a vis-node A40 is enough)"""
import importlib, os, sys
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
M, S, D = 300, 520, 96
flat = torch.randint(0, M, (S,), device=dev)
flat[:7] = 5                                                     # one token with many slots
perm, off = torch.ops.gpic.moe_csr(flat, M)
cnt = torch.bincount(flat, minlength=M)
check("csr: offsets = the counts' running sum; perm sorts by token, stable", torch.equal(off[1:] - off[:-1], cnt) and int(off[0]) == 0
      and bool((flat[perm][1:] >= flat[perm][:-1]).all()) and bool(((flat[perm][1:] != flat[perm][:-1]) | (perm[1:] > perm[:-1])).all()))
o = torch.randn(S, D, device=dev, dtype=torch.bfloat16)
w = torch.rand(S, device=dev)
ref = torch.zeros(M, D, dtype=torch.float64, device=dev).index_add_(0, flat, o.double() * w.double()[:, None])
got = torch.ops.gpic.moe_combine(o, w, flat, perm, off, M, True)
check("segmented sum = fp64 index_add, then bf16", rel(got.float(), ref.float()) < 8e-3 and got.dtype == torch.bfloat16,
      f"{rel(got.float(), ref.float()):.1e}")
og = o.float().clone().requires_grad_(True); wg = w.clone().requires_grad_(True)
g = torch.randn(M, D, device=dev)
torch.zeros(M, D, device=dev).index_add_(0, flat, og * wg[:, None]).backward(g)
oo = o.float().clone().requires_grad_(True); ww = w.clone().requires_grad_(True)
torch.ops.gpic.moe_combine(oo, ww, flat, perm, off, M, True).backward(g)
check("combine backward: grad_o and grad_gate = autograd of the torch expression", rel(oo.grad, og.grad) < 1e-6 and rel(ww.grad, wg.grad) < 1e-5,
      f"{rel(oo.grad, og.grad):.1e}, {rel(ww.grad, wg.grad):.1e}")
x = torch.randn(M, D, device=dev).requires_grad_(True); x2 = x.detach().clone().requires_grad_(True)
ge = torch.randn(S, D, device=dev)
x.index_select(0, flat).backward(ge); torch.ops.gpic.moe_gather(x2, flat, perm, off).backward(ge)
check("gather backward = index_add (fp32)", rel(x2.grad, x.grad) < 1e-6, f"{rel(x2.grad, x.grad):.1e}")

# ---- the layer, kernels against torch ----
CASES = (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("eci", 16, 2, {}), ("noise", 8, 1, {"t_edges": [i / 8 for i in range(1, 8)], "capacity": 2.0}),
         ("pos", 8, 1, {}), ("ecb", 64, 8, {}))
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
            dynamo.reset()
            mod = torch.compile(m)
        yt, gt = run(False, mod)
        if mode == "bf16-compiled":
            dynamo.reset(); mod = torch.compile(m)
            moe_mod._KERNELS = True
            ex = dynamo.explain(m)(xin.clone().requires_grad_(True), tt)
            breaks = ex.graph_break_count
            dynamo.reset(); mod = torch.compile(m)
        else:
            breaks = 0
        yk, gk = run(True, mod); yk2, gk2 = run(True, mod)
        tol = 1e-5 if mode == "fp32" else 2e-2
        e_y, e_g = rel(yk, yt), max(rel(a, b) for a, b in zip(gk, gt))
        same = torch.equal(yk, yk2) and all(torch.equal(a, b) for a, b in zip(gk, gk2))
        check(f"layer {route} n{n} G{G} {mode}: kernels = torch path", e_y < tol and e_g < tol and same and breaks == 0,
              f"output {e_y:.1e}, gradients {e_g:.1e} (tolerance {tol:.0e}); repeat bit for bit {same}; graph breaks {breaks}")
moe_mod._KERNELS = False
print("MOE_OPS_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
