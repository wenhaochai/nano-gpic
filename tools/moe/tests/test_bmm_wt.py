#!/usr/bin/env python3
"""GPIC_MOE_BMM_WT (src/models/layers/moe.py _BmmWT: the experts' first GEMM with its gradients in their own layouts) against autograd of
torch.bmm(xe, W.mT): the MoE layer (ecb 16 experts with 2 active, tc, and ecb 64 with 8), forward + backward, in fp32 (no autocast; within
1e-6 of the largest value: the same products, cuBLAS's order), bf16 autocast on fp32 parameters, bf16 parameters and input as FSDP2 trains
(within 2e-2), and bf16 under torch.compile (within 2e-2, no graph break). With the torch path and with the v3 operators.
usage: python test_bmm_wt.py   (a CUDA GPU; an A40 is enough)"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
import src.models.layers.moe as moe_mod   # noqa: E402
import torch._dynamo as dynamo           # noqa: E402
dev = "cuda"
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


rel = lambda a, b: float((a.double() - b.double()).abs().max()) / max(float(b.double().abs().max()), 1e-12)
d, T, B = 128, 256, 8
for kernels in (0, 3):
    for route, n, G, kw in (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("ecb", 64, 8, {})):
        for mode in ("fp32", "bf16", "bf16-params", "bf16-compiled"):
            torch.manual_seed(0)
            m = moe_mod.MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T, **kw).to(dev).train()
            torch.nn.init.normal_(m.router.weight, std=0.05)
            if mode == "bf16-params":
                for name, q in m.named_parameters():
                    if not name.startswith("router."):
                        q.data = q.data.to(torch.bfloat16)
            xin = torch.randn(B, T, d, device=dev, dtype=torch.bfloat16 if mode == "bf16-params" else torch.float32)
            tt = torch.rand(B, device=dev); gy = torch.randn(B, T, d, device=dev)

            def run(wt, mod):
                moe_mod._BMM_WT = wt; moe_mod._KERNELS = kernels
                xx = xin.clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32"):
                    y = mod(xx, tt)
                y.float().backward(gy)
                gs = [xx.grad] + [p.grad.clone() for p in m.parameters() if p.grad is not None]
                for p in m.parameters():
                    p.grad = None
                return y.detach().float(), gs
            mod = m
            breaks = 0
            if mode == "bf16-compiled":
                dynamo.reset(); mod = torch.compile(m)
            y0, g0 = run(False, mod)
            if mode == "bf16-compiled":
                dynamo.reset(); moe_mod._BMM_WT = True; moe_mod._KERNELS = kernels
                breaks = dynamo.explain(m)(xin.clone().requires_grad_(True), tt).graph_break_count
                dynamo.reset(); mod = torch.compile(m)
            y1, g1 = run(True, mod)
            tol = 1e-6 if mode == "fp32" else 2e-2
            e_y, e_g = rel(y1, y0), max(rel(a, b) for a, b in zip(g1, g0))
            check(f"kernels {kernels}, {route} n{n} G{G}, {mode}: _BmmWT = autograd of bmm(xe, W.mT)",
                  e_y < tol and e_g < tol and breaks == 0 and len(g1) == len(g0) and all(a.dtype == b.dtype for a, b in zip(g1, g0)),
                  f"output {e_y:.1e}, gradients {e_g:.1e} (tolerance {tol:.0e}); graph breaks {breaks}")
moe_mod._BMM_WT = False; moe_mod._KERNELS = 0
print("BMM_WT_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
