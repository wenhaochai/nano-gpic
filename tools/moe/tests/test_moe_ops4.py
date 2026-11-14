#!/usr/bin/env python3
"""GPU test of the MoE operators v4 (GPIC_MOE_KERNELS=4: the gate on the experts' hidden activations, the combine a plain sum) and of the
router's score operator (GPIC_MOE_ROUTER=1: gpic::moe_scores), src/models/layers/moe_ops.py.

The score operator alone: scores, the input's gradient and the weight's gradient against float64, with IEEE fp32 products (fp32 sums:
2e-6 of the largest value; a bf16 input's gradient: bf16 rounding) and with TF32 products (the default, as F.linear's GEMM on an H100:
2e-3), bf16 and fp32 inputs, shapes that do not fill the kernels' blocks, 1 to 128 experts; two calls bit for bit equal. The layer, every route (and the granularity runs' shapes: 128 experts with 16 active, 64 with 8), fp32 (no autocast),
bf16 autocast and bf16 under torch.compile, against the torch path with F.linear scores: v4, v4 with the score operator, and v3 with the
score operator (fp32 within 1e-5 of the largest value: the order of fp32 terms; bf16 within 2e-2); two calls bit for bit equal; no graph
break when compiled. Evaluation (tc and ecb route by threshold there) the same way. These layer checks run the score operator with IEEE
products, so that it selects the tokens the torch path selects; with TF32 products a last check counts the (token, expert) choices
that differ from the torch path's (under 0.1%).
usage: python test_moe_ops4.py   (a CUDA GPU; a vis-node A40 is enough)"""
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

# ---- the score operator alone ----
for M, D, N, prec in [(*c, q) for q in ("ieee", "tf32") for c in ((300, 96, 6), (2048, 128, 16), (8192, 896, 128), (1000, 1024, 4), (513, 80, 1))]:
    moe_ops._SC_PREC = prec
    for dt in (torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        x = torch.randn(M, D, device=dev).to(dt).requires_grad_(True)
        w = (torch.randn(N, D, device=dev) * 0.05).requires_grad_(True)
        g = torch.randn(M, N, device=dev)
        s = torch.ops.gpic.moe_scores(x, w)
        s.backward(g)
        s2 = torch.ops.gpic.moe_scores(x, w)
        xd, wd, gd = x.detach().double(), w.detach().double(), g.double()
        e_s, e_x, e_w = rel(s, xd @ wd.T), rel(x.grad, gd @ wd), rel(w.grad, gd.T @ xd)
        tol = 2e-6 if prec == "ieee" else 2e-3        # TF32: the operands rounded to 10 mantissa bits
        tol_x = tol if dt == torch.float32 else 4e-3   # a bf16 gradient: half a unit of 8 mantissa bits
        check(f"scores [{M}, {D}] x {N} experts, {str(dt).replace('torch.', '')} input, {prec} products",
              e_s < tol and e_x < tol_x and e_w < tol and torch.equal(s, s2) and s.dtype == torch.float32 and x.grad.dtype == dt
              and w.grad.dtype == torch.float32,
              f"scores {e_s:.1e}, input gradient {e_x:.1e} (tolerance {tol_x:.0e}), weight gradient {e_w:.1e}; repeat bit for bit {torch.equal(s, s2)}")

# ---- the layer against the torch path (kernels 0, F.linear scores) ----
moe_ops._SC_PREC = "ieee"   # the same token choices as the torch path (its F.linear is IEEE fp32 here: torch's default matmul precision)
CASES = (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("eci", 16, 2, {}), ("noise", 8, 1, {"t_edges": [i / 8 for i in range(1, 8)], "capacity": 2.0}),
         ("pos", 8, 1, {}), ("hash", 8, 1, {}), ("ecb", 64, 8, {}), ("ecb", 128, 16, {}), ("ecb", 16, 1, {}),
         ("noise", 16, 1, {"t_edges": [i / 16 for i in range(1, 16)], "capacity": 2.0}))
d, T, B = 128, 256, 8
ARMS = ((4, False, "v4"), (4, True, "v4 + score operator"), (3, True, "v3 + score operator"))
for route, n, G, kw in CASES:
    for mode in ("fp32", "bf16", "bf16-compiled"):
        torch.manual_seed(0)
        m = moe_mod.MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T, **kw).to(dev).train()
        if m.router is not None:
            torch.nn.init.normal_(m.router.weight, std=0.05)
        xin = torch.randn(B, T, d, device=dev); tt = torch.rand(B, device=dev); gy = torch.randn(B, T, d, device=dev)

        def run(kernels, router, mod):
            moe_mod._KERNELS, moe_mod._ROUTER = kernels, router
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
        yt, gt = run(0, False, mod)
        tol = 1e-5 if mode == "fp32" else 2e-2
        for K, R, name in ARMS:
            if R and m.router is None:
                continue
            breaks = 0
            if mode == "bf16-compiled":
                dynamo.reset()
                moe_mod._KERNELS, moe_mod._ROUTER = K, R
                breaks = dynamo.explain(m)(xin.clone().requires_grad_(True), tt).graph_break_count
                dynamo.reset(); mod = torch.compile(m)
            yk, gk = run(K, R, mod); yk2, gk2 = run(K, R, mod)
            e_y, e_g = rel(yk, yt), max(rel(a, b) for a, b in zip(gk, gt))
            same = torch.equal(yk, yk2) and all(torch.equal(a, b) for a, b in zip(gk, gk2))
            check(f"layer {route} n{n} G{G} {mode}: {name} = torch path", e_y < tol and e_g < tol and same and breaks == 0 and len(gk) == len(gt),
                  f"output {e_y:.1e}, gradients {e_g:.1e} (tolerance {tol:.0e}); repeat bit for bit {same}; graph breaks {breaks}")

# ---- evaluation (routes tc and ecb route by threshold there: a slot table as wide as the busiest expert, eager) ----
for route, n, G, kw in (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("ecb", 64, 8, {}), ("ecb", 128, 16, {})):
    for mode in ("fp32", "bf16"):
        torch.manual_seed(0)
        m = moe_mod.MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T, **kw).to(dev).eval()
        torch.nn.init.normal_(m.router.weight, std=0.05)
        if route == "ecb":   # measured thresholds, as a trained layer has (the boundary score of each expert's top tokens)
            m.moe_thr.copy_(torch.rand(n, device=dev) * 0.2 + 0.45); m.moe_thr_set.fill_(True)
        xin = torch.randn(B, T, d, device=dev); tt = torch.rand(B, device=dev)
        outs = []
        for k, r in ((0, False), (4, False), (4, False), (4, True), (3, True)):
            moe_mod._KERNELS, moe_mod._ROUTER = k, r
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32"):
                outs.append(m(xin, tt).float())
        tol = 1e-5 if mode == "fp32" else 2e-2
        errs = [rel(o, outs[0]) for o in outs[1:]]
        check(f"evaluation {route} n{n} G{G} {mode}: v4, v4 + score operator, v3 + score operator = torch path",
              max(errs) < tol and torch.equal(outs[1], outs[2]), f"{errs[0]:.1e}, {errs[2]:.1e}, {errs[3]:.1e}; repeat bit for bit {torch.equal(outs[1], outs[2])}")
# ---- TF32 scores (the default): the choices against the torch path's ----
moe_ops._SC_PREC = "tf32"
for route, n, G in (("ecb", 16, 2), ("tc", 16, 2), ("eci", 16, 2), ("ecb", 128, 16)):
    torch.manual_seed(0)
    m = moe_mod.MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T).to(dev).train()
    torch.nn.init.normal_(m.router.weight, std=0.05)
    xin = torch.randn(64, T, d, device=dev)
    sels = []
    for r in (False, True):
        moe_mod._KERNELS, moe_mod._ROUTER = 3, r
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            idx, gate = m.slots(xin)
        sel = torch.zeros(64 * T, n, dtype=torch.bool, device=dev)
        sel[idx.reshape(-1), torch.arange(n, device=dev).repeat_interleave(idx.shape[1])] = (gate.reshape(-1) != 0)
        sels.append(sel)
    diff = float((sels[0] != sels[1]).sum()) / float(sels[0].sum())
    check(f"choices {route} n{n} G{G}: TF32 score operator against F.linear", diff < 1e-3, f"{diff:.1e} of the (token, expert) pairs differ")
moe_mod._KERNELS, moe_mod._ROUTER = 0, False
print("MOE_OPS4_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
