#!/usr/bin/env python3
"""The noise route outside training (2026-11-15: the held-out loss of moe-rule-d11-noise-n8-a1 came out 0.1128, +68% over dense, while its
training loss was within 1.4% of the other MoE runs). Outside training every image of a batch shares one noise level (a held-out level,
a sampler's step), so all its tokens choose one expert; the training slot table's capacity (capacity x tokens / n) then dropped about
7/8 of them. Checks, CPU:
  1. evaluation, a batch of 16 images at one t equals the same 16 images one by one (one image alone always fits the capacity);
  2. evaluation, a batch spread over all bins equals the images one by one;
  3. no token is dropped: the batch at one t equals the expert applied to every token by hand;
  4. training is unchanged: the training forward of a spread batch equals the slot path of the code before the fix (same function).
usage: python test_noise_eval.py"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen")))
from src.models.layers.moe import MoEFeedForward          # noqa: E402

OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name} {detail}", flush=True)


torch.manual_seed(0)
n, d, T = 8, 64, 256
edges = [0.1 + 0.1 * i for i in range(n - 1)]
m = MoEFeedForward(d, 4 * d, n, active=1, route="noise", capacity=1.5, t_edges=edges, tokens=T)
for p in m.parameters():
    torch.nn.init.normal_(p, std=0.05)
B = 16
x = torch.randn(B, T, d)
one_t = torch.full((B,), 0.35)
spread = torch.linspace(0.02, 0.98, B)

m.eval()
with torch.no_grad():
    for name, t in (("one t", one_t), ("spread t", spread)):
        y = m(x, t=t)
        ref = torch.cat([m(x[i:i + 1], t=t[i:i + 1]) for i in range(B)])
        err = (y - ref).abs().max().item()
        check(f"eval, batch = image by image ({name})", err < 1e-5, f"max |diff| {err:.2e}")
    # 3. by hand: every token of every image through its bin's expert
    e = int((0.35 > torch.tensor(edges)).sum())
    y = m(x, t=one_t)
    sel = torch.zeros(B * T, n, dtype=torch.bool); sel[:, e] = True
    idx, gate, _ = MoEFeedForward._compact(sel, B * T)
    hand = m._experts(x.reshape(B * T, d), idx, gate).view(B, T, d)
    if m.shared is not None:
        hand = hand + m.shared(x)
    err = (y - hand).abs().max().item()
    zero_rows = int((y.reshape(B * T, d).abs().sum(1) == 0).sum())
    check("eval at one t: no token dropped", err < 1e-5 and zero_rows == 0, f"max |diff| {err:.2e}, zero outputs {zero_rows}")

m.train()
with torch.no_grad():
    y = m(x, t=spread)
    idx, gate = m.slots(x, spread)
    ref = m._experts(x.reshape(B * T, d), idx, gate).view(B, T, d)
    if m.shared is not None:
        ref = ref + m.shared(x)
    err = (y - ref).abs().max().item()
    check("training forward = the training slot table (unchanged path)", err == 0.0, f"max |diff| {err:.2e}")
print("ALL OK" if OK else "SOME FAILED")
sys.exit(0 if OK else 1)
