#!/usr/bin/env python3
"""MFU lever MA (src/optim/muon_add.py): Muon's weight updates in one Triton launch against torch's add_ per part, bit for
bit: (1) random parts (fp32 rows of matrices, bf16 updates, varied lengths and alphas); (2) a whole MuonAdamW step of a
small model, the switch on against off (the same weights after 3 steps). One GPU.
usage: python test_muon_add.py"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
import src.optim.muon as MU  # noqa: E402
from src.optim.muon_add import add_units_triton  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
W = [torch.randn(r, c, device=dev) for r, c in [(1536, 1536), (6144, 1536), (37, 640), (1, 4096), (3000, 1280)]]
D = [w[3:] if w.shape[0] > 3 else w for w in W]
S = [torch.randn(d.shape, device=dev).to(torch.bfloat16) for d in D]
A = [-3.4e-3 * (k + 1) * 0.37 for k in range(len(D))]
ref = [d.clone() for d in D]
for r, s, a in zip(ref, S, A):
    r.add_(s, alpha=a)
add_units_triton(D, S, A)
g1 = all(torch.equal(d, r) for d, r in zip(D, ref))
print(f"1. {len(D)} parts, one launch = add_ per part bit for bit: {g1}", flush=True)


def steps(fused):
    MU._MUON_ADD = fused
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(640, 1920, bias=False), torch.nn.Linear(1920, 640, bias=False)).to(dev)
    for p in net.parameters():
        p._muon_parts = 1
    opt = MU.MuonAdamW(list(net.parameters()), lr=1e-3, muon_lr=3.4e-3, weight_decay=0.1)
    for k in range(3):
        x = torch.randn(256, 640, device=dev)
        net(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in net.parameters()]


a, b = steps(False), steps(True)
g2 = all(torch.equal(x, y) for x, y in zip(a, b))
print(f"2. MuonAdamW, 3 steps, the switch on = off bit for bit: {g2}", flush=True)
print("MUON_ADD_TEST " + ("PASS" if g1 and g2 else "FAIL"))
sys.exit(0 if g1 and g2 else 1)
