#!/usr/bin/env python3
"""GPU check of the MoE layer under torch.compile (Inductor's CUDA codegen, which the CPU check does not reach: the first version's
cumsum over the tokens hit 'NYI TritonSplitDimKernel reductions' on CUDA, 2026-11-13). Every route, bf16 autocast, training forward +
backward twice (no recompile on the second call: torch._dynamo's counters), then evaluation; the compiled output and gradients match the
eager ones within bf16 rounding. Runs on any CUDA GPU (a vis node's A40 for the codegen, an H100 in the smoke).
usage: python test_moe_compile_gpu.py [--d 256 --images 16]"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.layers.moe import MoEFeedForward   # noqa: E402
import torch._dynamo.utils as du   # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--d", type=int, default=256); p.add_argument("--images", type=int, default=16)
a = p.parse_args()
dev, T, d = "cuda", 256, a.d
ok = True
for route, n, G, kw in (("ecb", 16, 2, {}), ("tc", 16, 2, {"capacity": 1.5}), ("eci", 16, 2, {}), ("noise", 8, 1, {"t_edges": [i / 8 for i in range(1, 8)], "capacity": 2.0}),
                        ("pos", 8, 1, {}), ("hash", 8, 1, {}), ("ecb", 64, 8, {}), ("tc", 8, 1, {"capacity": 1.0})):
    torch.manual_seed(0)
    m = MoEFeedForward(d, 4 * d, experts=n, active=G, route=route, tokens=T, **kw).to(dev).train()
    if m.router is not None:
        torch.nn.init.normal_(m.router.weight, std=0.05)
    x = torch.randn(a.images, T, d, device=dev); t = torch.rand(a.images, device=dev)
    def step(mod):
        xx = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(xx, t)
        y.float().square().mean().backward()
        g = [xx.grad] + [p.grad.clone() for p in m.parameters() if p.grad is not None]
        for p in m.parameters():
            p.grad = None
        return y.detach().float(), g
    ye, ge = step(m)
    torch._dynamo.reset(); du.counters.clear()
    c = torch.compile(m)
    yc, gc = step(c); yc2, _ = step(c)
    rec = sum(du.counters["recompiles"].values()) if "recompiles" in du.counters else 0
    err = float((yc - ye).abs().max()) / max(float(ye.abs().max()), 1e-6)
    gerr = max(float((u - v).abs().max()) / max(float(v.abs().max()), 1e-6) for u, v in zip(gc, ge))
    m.eval(); c.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ev, evc = m(x, t).float(), c(x, t).float()
    m.train()
    eerr = float((evc - ev).abs().max()) / max(float(ev.abs().max()), 1e-6)
    rep = float((yc - yc2).abs().max()) / max(float(yc.abs().max()), 1e-6)   # fp32 atomics: the order of a token's terms may change the last bf16 bit
    good = err < 0.03 and gerr < 0.05 and eerr < 0.03 and rep < 0.01 and rec == 0
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} {route} n{n} G{G}: compiled vs eager, output {err:.1e}, gradients {gerr:.1e} (relative to the largest), "
          f"evaluation {eerr:.1e}; second call {rep:.1e}; recompiles {rec}", flush=True)
print("MOE_COMPILE_GPU " + ("PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)
