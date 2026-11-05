#!/usr/bin/env python3
"""The metrics of tools/ar/follow_ar.py against draws whose answer is known: run from the repo root, no GPU."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from follow_ar import baseline, effective, measure

torch.manual_seed(0)
N, T, P, B = 16, 64, 12, 8
# Two families of 8 images. In the first half of the positions, images of a family share a patch up to small noise, so a few known tokens there name only the family. In the second half, each image has its own patch.
family = torch.randn(2, T, P)
true = family[torch.arange(N) % 2] + 0.01 * torch.randn(N, T, P)
true[:, T // 2:] = torch.randn(N, T // 2, P)
KS = (0, 1, 2, 4, 8, 16, 32)


def oracle(tokens, pick):
    """The exact chain rule at this tolerance: one uniform image among those that agree with the known tokens."""
    out = tokens.clone()
    for r in range(len(tokens)):
        agree = (((true - tokens[r]) ** 2).mean(-1).sqrt() < 0.2)[:, ~pick[r]].all(-1).nonzero().flatten()
        b = agree[torch.randint(len(agree), (1,))]
        out[r, pick[r]] = true[b, pick[r]].squeeze(0) if pick[r].sum() else out[r, pick[r]]
    return out


def ignores(tokens, pick):
    """Each drawn token comes from its own uniform image, whatever is known."""
    images = torch.randint(0, N, (len(tokens), T))
    return torch.where(pick[..., None], true[images, torch.arange(T)], tokens)


def blur(tokens, pick):
    return torch.where(pick[..., None], true.mean(0).expand_as(tokens), tokens)


res, counts = measure(oracle, true, KS, repeats=6, batch=B)
for k, r in res.items():
    se = (r["follow"] * (1 - r["follow"]) / (6 * N)) ** 0.5   # one image per case, so the case is the unit
    assert abs(r["follow"] - r["ideal"]) < 4 * max(se, 0.01), (k, r)
    assert r["manifold"] == 1, (k, r)
assert 0.2 < res[0]["ideal"] < 0.3, res[0]   # a uniform image: 1/2 of the families at half the positions, 1/16 at the rest: 0.28
assert res[32]["ideal"] > 0.99 and res[1]["ideal"] < res[8]["ideal"]
print("oracle follows the chain rule:", {k: round(r["follow"], 3) for k, r in res.items()}, "ideal", {k: round(r["ideal"], 3) for k, r in res.items()})
chance = res[0]["ideal"]
res_i, counts_i = measure(ignores, true, KS, repeats=6, batch=B)
for k, r in res_i.items():
    assert abs(r["follow"] - chance) < 0.04 and r["manifold"] == 1, (k, r)
assert res_i[32]["ideal"] > 0.99
print("a draw that ignores the known tokens stays at chance:", {k: round(r["follow"], 3) for k, r in res_i.items()})
res_b, _ = measure(blur, true, (0, 8), repeats=2, batch=B)
assert res_b[0]["manifold"] < 0.02 and res_b[0]["follow"] < 0.02
print("a blur is not a real patch:", round(res_b[0]["manifold"], 3))
res_c, _ = measure(lambda t, p: t, true, (0, 8), repeats=2, batch=B)
assert res_c[8]["follow"] == 1 and res_c[8]["own_rms"] == 0
print("the image itself: follow 1")
base = baseline(true, 0.03, int(counts_i[0].sum().item()), B)
ratio = effective(counts_i).mean().item() / effective(base).mean().item()
assert 0.9 < ratio < 1.1, ratio
single, _ = None, None
_, counts_one = measure(lambda t, p: torch.where(p[..., None], true[:1].expand_as(t), t), true, (0,), repeats=1, batch=B)
assert (effective(counts_one) - 1).abs().max() < 1e-4 and abs(effective(counts_one.sum(0)).item() - 1) < 1e-4
print("effective images: uniform against its baseline", round(ratio, 3), "one image", 1)
try:
    measure(lambda t, p: t + 1, true, (1,), repeats=1, batch=B)
except AssertionError as e:
    assert "known token" in str(e)
    print("a changed known token stops the run")
else:
    raise SystemExit("a changed known token was not caught")
print("all follow checks passed")
