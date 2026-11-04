#!/usr/bin/env python3
"""CPU checks of plan A's code. Run on a vis node from baselines/PixelGen with PYTHONPATH=. : python ../../tools/dist_step/test_code.py"""
import math

import torch

from src.models.transformer.JiT_T2I import JiT_T2I
from src.diffusion.flow_matching.training_code import CodeJiTTrainer
from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant
from src.diffusion.flow_matching.scheduling import LinearScheduler

kw = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=16, bottleneck_dim=16)
trainer = lambda cls, **k: cls(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, null_condition_p=0.0, **k)
x = torch.randn(4, 3, 64, 64)
t = torch.rand(4)
y = torch.randn(4, 8, 32)
mask = torch.ones(4, 8, dtype=torch.bool)
mask[:, 6:] = False

# 1. code_dim = 0 is the point model: the same parameters and outputs as JiT_T2I without the arguments, and the same loss as JiTTrainer on the same RNG stream.
torch.manual_seed(1)
a = JiT_T2I(**kw)
torch.manual_seed(1)
b = JiT_T2I(**kw, code_dim=0)
assert list(dict(a.named_parameters())) == list(dict(b.named_parameters()))
assert all((p == q).all() for p, q in zip(a.parameters(), b.parameters()))
assert (a(x, t, y, y_mask=mask) == b(x, t, y, y_mask=mask)).all()
for net in (a,):
    net.final_layer.linear.weight.data.normal_(std=0.05)
torch.manual_seed(5)
l1 = trainer(JiTTrainer)._impl_trainstep(a, None, None, x, y, {"y_mask": mask})
torch.manual_seed(5)
l2 = trainer(CodeJiTTrainer)._impl_trainstep(a, None, None, x, y, {"y_mask": mask})
assert l1["loss"].item() == l2["loss"].item() and "prior_loss" not in l2, (l1["loss"].item(), l2["loss"].item())

# 2. The channel: the power and the noise, for each rate, and the bound's formula.
torch.manual_seed(2)
c = JiT_T2I(**kw, code_dim=8, code_rate=4.0)
for rate in (1.0, 4.0, 16.0, 64.0):
    c.code_rate = rate
    a2 = 1 - math.exp(-2 * rate / 8)
    assert abs(-8 / 2 * math.log(1 - a2) - rate) < 1e-9
    same = x[:1].repeat(2000, 1, 1, 1)
    z = c.code_encode(same)
    assert abs(z.var(0).mean().item() - (1 - a2)) < 0.05, (rate, z.var(0).mean().item(), 1 - a2)   # one image: only the noise is left
    assert abs(z.mean(0).pow(2).sum().item() - a2 * 8) < 0.1 * 8, (rate, z.mean(0).pow(2).sum().item(), a2 * 8)   # its centre lies on the sphere
assert c.code_encode(torch.zeros(4, 3, 64, 64)).isfinite().all(), "a zero image in the warm-up must stay finite"

# 3. z reaches the output, and the gradient reaches every parameter through the two losses.
c.code_rate = 4.0
c.final_layer.linear.weight.data.normal_(std=0.05)
for blk in list(c.blocks) + list(c.text_refine_blocks):   # the conditioning is zero at init, by the official init, so z acts only after the first update
    blk.adaLN_modulation[-1].weight.data.normal_(std=0.05)
c.code_prior.out.weight.data.normal_(std=0.05)
z = c.code_encode(x)
assert (c(x, t, y, y_mask=mask, z=z) - c(x, t, y, y_mask=mask, z=-z)).abs().max() > 1e-5
tr = trainer(CodeJiTTrainer)
torch.manual_seed(6)
c.zero_grad()
res = tr._impl_trainstep(c, None, None, x, y, {"y_mask": mask})
res["loss"].backward()
none = [n for n, p in c.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
assert not none, none

# 4. The flow does not change the code: without the main loss, the encoder gets no gradient.
c.zero_grad()
c.code_prior_loss(c.code_encode(x).detach(), y, mask).backward()
assert all(p.grad is None for p in c.code_enc.parameters()) and all(p.grad is None for p in c.blocks.parameters())
assert any(p.grad is not None for p in c.code_prior.parameters())

# 5. The flow learns a shifted, narrow law given the caption: its samples have the mean and the variance of the data.
torch.manual_seed(7)
f = JiT_T2I(**kw, code_dim=4, code_rate=4.0)
opt = torch.optim.AdamW(f.code_prior.parameters(), lr=1e-3)
cap = torch.randn(1, 8, 32).repeat(256, 1, 1)
m = torch.ones(256, 8, dtype=torch.bool)
for _ in range(1500):
    data = 1.5 + 0.3 * torch.randn(256, 4)
    loss = f.code_prior_loss(data, cap, m)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    s = f.code_flow(torch.randn(256, 4), cap, m)
assert abs(s.mean().item() - 1.5) < 0.1 and abs(s.std().item() - 0.3) < 0.1, (s.mean().item(), s.std().item())

# 6. code_sample: the same noise gives the same z, another noise another z, and the start is 1 in N(0, 1) per dimension.
n = torch.randn(4, 3, 64, 64)
assert (f.code_sample(n, y, mask) == f.code_sample(n.clone(), y, mask)).all()
assert (f.code_sample(n, y, mask) - f.code_sample(torch.randn(4, 3, 64, 64), y, mask)).abs().max() > 1e-4
flat = torch.randn(1000, 3, 256, 256).flatten(1)
d = 64
start = flat[:, :: flat.shape[1] // d][:, :d]
assert start.shape == (1000, d) and abs(start.std().item() - 1) < 0.05

# 7. FLOPs of the new parts at d640, in the counting of tools/ladder_flops.py (2 per multiply-add).
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
big = dict(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
with torch.device("meta"):
    g = JiT_T2I(**big, code_dim=64, code_rate=64.0)
with FlopCounterMode(display=False) as fc:
    g.code_enc(torch.zeros(1, 256, 768, device="meta"))
enc = fc.get_total_flops()
with FlopCounterMode(display=False) as fc:
    g.code_prior(torch.zeros(1, 64, device="meta"), torch.zeros(1, device="meta"), g.code_prior.y_proj(torch.zeros(1, 2048, device="meta")))
flow = fc.get_total_flops()
print(f"all checks passed; encoder {enc / 1e6:.0f} MFLOP, flow {flow / 1e6:.0f} MFLOP per image; forward 34.7 GFLOP: {100 * (enc + flow) / 34.7e9:.2f}%")
print("parameters of the new parts (M):", {n: round(sum(p.numel() for p in getattr(g, n).parameters()) / 1e6, 3) for n in ("code_enc", "code_proj", "code_prior")})
