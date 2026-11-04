#!/usr/bin/env python3
"""CPU checks of the autoregressive model. Run on a vis node from baselines/PixelGen with PYTHONPATH=. and GPIC_AR_NO_COMPILE=1 : python ../../tools/ar/test_ar.py"""
import torch

from src.diffusion.flow_matching.training_ar import ARTrainer
from src.models.transformer.AR_JiT import ARJiT

kw = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=16, bottleneck_dim=16, head_width=64, head_depth=2, head_steps=8)
torch.manual_seed(0)
net = ARJiT(**kw)
N, P = net.num_patches, net.patch_dim
assert (N, P) == (16, 768)
for blk in list(net.blocks) + list(net.text_refine_blocks):   # the official init makes the blocks the identity; here they must act
    blk.adaLN_modulation[-1].weight.data.normal_(std=0.05)
    blk.adaLN_modulation[-1].bias.data.normal_(std=0.2)
net.head.out.weight.data.normal_(std=0.05)
for blk in net.head.blocks:
    blk.adaLN_modulation[-1].weight.data.normal_(std=0.05)
    blk.adaLN_modulation[-1].bias.data.normal_(std=0.2)

tokens = torch.randn(2, N, P)
y = torch.randn(2, 8, 32)
mask = torch.ones(2, 8, dtype=torch.bool)
mask[:, 6:] = False

# 1. Causality: z_i does not change with patch j >= i, and does change with some patch j < i.
z = net(tokens, y, y_mask=mask)
for j in (0, 5, 15):
    changed = tokens.clone()
    changed[:, j] += 1.0
    d = (net(changed, y, y_mask=mask) - z).abs().amax(dim=(0, 2))
    assert (d[: j + 1] == 0).all(), (j, d[: j + 1])
    if j < N - 1:
        assert (d[j + 1:] > 1e-5).all(), (j, d[j + 1:])

# 2. The caption reaches z at every position, the first included; padded words do not; and with no mask given all words count.
other = y.clone()
other[:, 0] += 1.0
assert (net(tokens, other, y_mask=mask) - z).abs().amax(dim=(0, 2)).min() > 1e-5
pad = y.clone()
pad[:, 6:] += 1.0
assert (net(tokens, pad, y_mask=mask) == z).all()
full, full2 = torch.randn(2, 16, 32), None   # the official unmasked path: captions padded to the maximum length
full2 = full.clone()
full2[:, 15] += 1.0
assert (net(tokens, full2) != net(tokens, full)).any()

# 3. The loss: the head's shape and the official formula, and a gradient on every parameter.
trainer = ARTrainer(P_std=1.6, null_condition_p=0.0)
x = torch.randn(2, 3, 64, 64)
torch.manual_seed(3)
res = trainer._impl_trainstep(net, None, None, x, y, {"y_mask": mask})
assert res["loss"].isfinite() and res["loss_per_sample"].shape == (2,)
net.zero_grad()
res["loss"].backward()
none = [n for n, p in net.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
assert not none, none

# 4. Teacher forcing and sampling agree: the patch order and the fold are inverse of unfold.
img = torch.arange(3 * 64 * 64, dtype=torch.float32).view(1, 3, 64, 64)
p = torch.nn.functional.unfold(img, kernel_size=16, stride=16).transpose(1, 2)
assert (net.unpatchify(p, 64, 64) == img).all()

# 5. The head alone learns a distribution with two modes: the samples must split by the true shares, and each mode must be sharp.
torch.manual_seed(5)
small = ARJiT(**{**kw, "head_width": 256, "head_depth": 3})   # a real head is wider; a width of 64 gets the shares wrong (tools/ar/head_modes.py)
small.head_steps = 32
opt = torch.optim.AdamW(small.head.parameters(), lr=1e-3)
a, b = torch.ones(768) * 0.6, -torch.ones(768) * 0.5
c = torch.zeros(1, 64)
for _ in range(1500):
    m = torch.rand(256) < 0.7
    target = torch.where(m[:, None], a, b)
    tau = (torch.randn(256) * 1.6 - 0.8).sigmoid()
    eta = torch.randn(256, 768)
    q = tau[:, None] * target + (1 - tau[:, None]) * eta
    hat = small.head(q, tau, c.repeat(256, 1))
    loss = (((hat - target) / (1 - tau[:, None]).clamp_min(0.05)) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
draws = small.draw_patch(c.repeat(2000, 1))
share = (draws.mean(1) > 0).float().mean().item()
assert abs(share - 0.7) < 0.08, share
assert (draws[draws.mean(1) > 0] - a).abs().mean() < 0.1 and (draws[draws.mean(1) < 0] - b).abs().mean() < 0.1

# 6. Overfit a few small images with the real trainer: the chain must return them. This tests the whole path, on CPU.
torch.manual_seed(6)
few = ARJiT(**kw)
imgs = torch.stack([torch.rand(3, 1, 1).repeat(1, 64, 64) * 2 - 1 for _ in range(4)])   # flat colours
null = torch.zeros(8, 2, 32)
null_mask = torch.ones(8, 2, dtype=torch.bool)
opt = torch.optim.AdamW(few.parameters(), lr=1e-3)
for step in range(1200):
    batch = imgs[torch.randint(4, (8,))]
    loss = trainer._impl_trainstep(few, None, None, batch, null, {"y_mask": null_mask})["loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()
few.head_steps = 16
out = few.ar_sample(null, null_mask)
nn = torch.cdist(out.flatten(1), imgs.flatten(1)).min(1)[0] / (3 * 64 * 64) ** 0.5
print("overfit on 4 flat images: rms to the nearest image per sample", [round(v, 3) for v in nn.tolist()], "last loss", round(loss.item(), 4))
assert nn.mean() < 0.25, nn

# 7. FLOPs at d640, in the counting of tools/ladder_flops.py (2 per multiply-add).
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
big = dict(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
with torch.device("meta"):
    g = ARJiT(**big)
with FlopCounterMode(display=False) as fc:
    g.head(torch.zeros(256, 768, device="meta"), torch.zeros(256, device="meta"), torch.zeros(256, 640, device="meta"))
head = fc.get_total_flops()
fwd = 34.72e9 - 2 * 256 * 640 * 768
total = fwd + head
print(f"all checks passed; head {head / 1e9:.2f} GFLOP per image, {100 * head / fwd:.1f}% of the backbone's {fwd / 1e9:.2f} GFLOP; training {3 * total / 1e9:.2f} GFLOP per image against 104.15")
print(f"parameters (M): backbone {sum(p.numel() for n, p in g.named_parameters() if not n.startswith('head.')) / 1e6:.1f}, head {sum(p.numel() for p in g.head.parameters()) / 1e6:.2f}")
print(f"steps at the compute of 66816 point steps (C0/8): {66816 * 104.15e9 / (3 * total):.0f}")
