#!/usr/bin/env python3
"""GPIC_MOE_EXPERT_WD (2026-11-15, owner popup 「专家权重衰减 ×1/8」): a factor on the MoE experts' decoupled weight decay in MuonAdamW. With zero
gradients one optimizer step is decay alone (Muon's orthogonalized zero is zero), so after one step every Muon matrix must be its old value times
1 - lr x wd x f, with f the factor for the experts' e12 and e3t and 1 for every other Muon matrix (here the block's dense projection); unset, f = 1
everywhere (the code before the change). CPU. usage: python test_expert_wd.py"""
import os, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.layers.moe import MoEFeedForward          # noqa: E402
import src.optim.muon as MU                                 # noqa: E402

OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = MoEFeedForward(d, 4 * d, 16, active=2, route="tc", capacity=1.5, tokens=16)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d)])


def one_step(factor):
    if factor is None:
        os.environ.pop("GPIC_MOE_EXPERT_WD", None)
    else:
        os.environ["GPIC_MOE_EXPERT_WD"] = str(factor)
    torch.manual_seed(0)
    net = Net(64)
    n, _ = MU.tag_muon_params(net)
    lr, wd = 1e-2, 0.05
    opt = MU.MuonAdamW(net.parameters(), lr=1e-3, weight_decay=0.0, muon_lr=lr, muon_weight_decay=wd)
    before = {k: v.detach().clone() for k, v in net.named_parameters()}
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    ratios = {}
    for k, v in net.named_parameters():
        if hasattr(v, "_muon_parts"):
            r = (v.detach() / before[k]).flatten()
            ratios[k] = (float(r.min()), float(r.max()))
    return n, ratios, 1 - lr * wd


for factor in (None, 0.125):
    n, ratios, base = one_step(factor)
    f = 1.0 if factor is None else factor
    want_e, want_d = 1 - (1 - base) * f, base
    exp = [r for k, r in ratios.items() if k.endswith((".e12", ".e3t"))]
    den = [r for k, r in ratios.items() if not k.endswith((".e12", ".e3t"))]
    check(f"factor {factor}: {n} Muon matrices, {len(exp)} expert, {len(den)} other", len(exp) == 4 and len(den) == 2)
    check(f"factor {factor}: experts decayed by 1 - lr wd x {f}", all(abs(a - want_e) < 1e-6 and abs(b - want_e) < 1e-6 for a, b in exp),
          f"{exp[0]} want {want_e:.6f}")
    check(f"factor {factor}: other Muon matrices decayed by 1 - lr wd", all(abs(a - want_d) < 1e-6 and abs(b - want_d) < 1e-6 for a, b in den),
          f"{den[0]} want {want_d:.6f}")
print("ALL OK" if OK else "SOME FAILED")
sys.exit(0 if OK else 1)
