#!/usr/bin/env python3
"""GPIC_MUON_NS_COMPILE (src/optim/muon.py): msign through torch.compile against the eager Newton-Schulz, on CUDA. For the batch shapes of
one rank (tall expert units, wide ones, square attention parts, 2-D), bf16, 5 steps, fused: the compiled result's error against an fp32
msign is at most 1.2 x the eager result's (both round in bf16; only the GEMM kernels differ), it is orthogonal to the same degree
(singular values within the eager result's range), and a second call is bit for bit the first. Without the switch msign is the eager code.
usage: GPIC_MUON_NS_COMPILE=max-autotune-no-cudagraphs python test_ns_compile.py   (a CUDA GPU; an A40 is enough)"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
os.environ.setdefault("GPIC_MUON_NS_COMPILE", "max-autotune-no-cudagraphs")
import src.optim.muon as mu   # noqa: E402
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


print("mode", mu._NS_COMPILE, torch.cuda.get_device_name())
torch.manual_seed(0)
for shape in ((12, 512, 256), (10, 256, 512), (6, 256, 256), (384, 128)):
    G = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    f32 = mu._msign_impl(G.float(), 5, torch.float32, True)
    eager = mu._msign_impl(G, 5, torch.bfloat16, True).float()
    comp = mu.msign(G, steps=5, dtype=torch.bfloat16, fused=True).float()
    comp2 = mu.msign(G, steps=5, dtype=torch.bfloat16, fused=True).float()
    big = float(f32.abs().max())
    e_e, e_c = float((eager - f32).abs().max()) / big, float((comp - f32).abs().max()) / big
    sv = lambda X: torch.linalg.svdvals(X.reshape(-1, *X.shape[-2:]).double())
    s_e, s_c = sv(eager).sort(dim=-1).values, sv(comp).sort(dim=-1).values
    ds = float((s_c - s_e).abs().max())   # absolute: a random square matrix keeps singular values near 0 after 5 steps, in both
    check(f"{tuple(shape)}: compiled error vs fp32 <= 1.2 x eager's; singular values within 0.05 of eager's; repeat equal",
          e_c <= 1.2 * e_e + 1e-6 and ds <= 0.05 and torch.equal(comp, comp2),
          f"errors {e_c:.2e} vs {e_e:.2e}; singular values [{float(s_c.min()):.3f}, {float(s_c.max()):.3f}] vs [{float(s_e.min()):.3f}, {float(s_e.max()):.3f}], max |diff| {ds:.3f}")
print("NS_COMPILE_TEST " + ("PASS" if OK else "FAIL")); sys.exit(0 if OK else 1)
