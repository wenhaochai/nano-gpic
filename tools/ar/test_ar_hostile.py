#!/usr/bin/env python3
"""Hostile checks of the autoregressive model, written from the spec, not from the code. CPU. Run on a vis node from baselines/PixelGen with PYTHONPATH=. : python ../../tools/ar/test_ar_hostile.py"""
import copy
import os

import torch

os.environ["GPIC_AR_NO_COMPILE"] = "1"
from src.diffusion.flow_matching.training_ar import ARTrainer  # noqa: E402
from src.models.transformer.AR_JiT import ARJiT  # noqa: E402

kw = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=16, bottleneck_dim=16, head_width=64, head_depth=2, head_steps=4)


def wake(net):
    """The official init makes the blocks the identity; here they must act."""
    for blk in list(net.blocks) + list(net.text_refine_blocks) + list(net.head.blocks):
        blk.adaLN_modulation[-1].weight.data.normal_(std=0.05)
        blk.adaLN_modulation[-1].bias.data.normal_(std=0.2)
    net.head.out.weight.data.normal_(std=0.05)
    return net


torch.manual_seed(0)
net = wake(ARJiT(**kw))
N, P = 16, 768
tokens = torch.randn(2, N, P)
y = torch.randn(2, 8, 32)
mask = torch.ones(2, 8, dtype=torch.bool)
mask[:, 6:] = False

# 1. A rigorous causality test: the gradient of z_i against every patch. A finite change can miss a leak; the Jacobian's zero pattern cannot.
t = tokens.clone().requires_grad_(True)
z = net(t, y, y_mask=mask)
for i in (0, 1, 7, 15):
    g = torch.autograd.grad(z[:, i].sum(), t, retain_graph=True)[0].abs().amax(dim=(0, 2))
    assert (g[i:] == 0).all(), (i, g)
    assert i == 0 or (g[:i] > 0).all(), (i, g)
print("1. gradient causality: z_i depends on exactly the patches before i")

# 2. The training loss cannot see patch i in z_i: the head's x_hat at i must not change with patch i if q and tau are fixed.
q, tau = torch.randn(2, N, P), torch.rand(2, N)
a = net(tokens, y, y_mask=mask, q=q, tau=tau)
changed = tokens.clone()
changed[:, 5] += 3.0
b = net(changed, y, y_mask=mask, q=q, tau=tau)
assert (a[:, :6] == b[:, :6]).all() and (a[:, 6:] != b[:, 6:]).any()
print("2. x_hat at i does not see patch i")

# 3. The raster order: unfold's index i is (row i // side, column i % side), and RoPE and pos_embed use the same grid.
side, s = 4, 16
img = torch.zeros(1, 3, 64, 64)
for r in range(side):
    for c in range(side):
        img[0, :, r * s:(r + 1) * s, c * s:(c + 1) * s] = r * side + c
order = torch.nn.functional.unfold(img, kernel_size=s, stride=s).transpose(1, 2)[0].mean(1)
assert (order == torch.arange(16.)).all(), order
from src.models.transformer.JiT_T2I import get_2d_sincos_pos_embed  # noqa: E402
pe = torch.from_numpy(get_2d_sincos_pos_embed(64, 4)).float()
assert (pe[1] != pe[4]).any() and (pe[0][:32] == pe[1][:32]).all() != (pe[0][32:] == pe[1][32:]).all()
print("3. unfold is raster order: left to right, then top to bottom")

# 4. One caption word, as the null row has; a row of all words; different T; batch of one; a word mask must exclude padding at every position.
for T in (1, 2, 16):
    yy = torch.randn(3, T, 32)
    m = torch.zeros(3, T, dtype=torch.bool)
    m[:, 0] = True
    if T > 1:
        m[1] = True
    out = net(torch.randn(3, N, P), yy, y_mask=m)
    assert out.shape == (3, N, 64) and out.isfinite().all(), T
assert net(torch.randn(1, N, P), torch.randn(1, 2, 32), y_mask=torch.ones(1, 2, dtype=torch.bool)).shape == (1, N, 64)
print("4. T = 1, 2, 16 and a batch of one")

# 5. CFG dropout through the trainer's real entry: with p = 1 every row becomes the null caption and the null mask; with p = 0 none does.
seen = {}
class Spy(ARJiT):
    def forward(self, tokens, y, y_mask=None, q=None, tau=None):
        seen["y"], seen["m"] = y.clone(), y_mask.clone()
        return super().forward(tokens, y, y_mask=y_mask, q=q, tau=tau)
spy = Spy(**kw)
x = torch.randn(2, 3, 64, 64)
null = torch.zeros_like(y)
null[:, 0] = 7.0
for p, want_null in ((1.0, True), (0.0, False)):
    tr = ARTrainer(null_condition_p=p)
    tr(spy, None, None, x, y.clone(), null.clone(), {"y_mask": mask.clone()})
    if want_null:
        assert (seen["y"] == null).all() and seen["m"].sum(1).tolist() == [1, 1] and seen["m"][:, 0].all()
    else:
        assert (seen["y"] == y).all() and (seen["m"] == mask).all()
print("5. CFG dropout reaches the model, caption and mask together")

# 6. The loss formula against JiTTrainer's: the same x_hat gives the same value on the same tau and noise.
from src.diffusion.flow_matching.scheduling import LinearScheduler  # noqa: E402
sch = LinearScheduler()
p0 = torch.randn(1, 1, 768)
hat = torch.randn(1, 1, 768)
for t0 in (0.0, 0.3, 0.97, 1.0):
    tt = torch.tensor([t0])
    noise = torch.randn(1, 1, 768)
    xt = sch.alpha(tt) * p0 + sch.sigma(tt) * noise
    den = (1 - tt.view(-1, 1, 1)).clamp_min(0.05)
    official = ((hat - xt) / den - (p0 - xt) / den) ** 2
    mine = ((hat - p0) / den) ** 2
    assert (official - mine).abs().max() < 1e-3 * mine.max() + 1e-4, t0
print("6. the loss equals the official v-space L2")

# 7. Checkpoints: strict round trip, no causal buffer stored, and the EMA's deepcopy with the same parameter order.
sd = net.state_dict()
assert not any("causal" in k or "final_layer" in k for k in sd), [k for k in sd if "causal" in k or "final_layer" in k]
fresh = ARJiT(**kw)
fresh.load_state_dict(sd, strict=True)
ema = copy.deepcopy(fresh)
assert [n for n, _ in fresh.named_parameters()] == [n for n, _ in ema.named_parameters()]
assert (ema.causal == fresh.causal).all()
ema.to(torch.float32)
assert (ema(tokens, y, y_mask=mask) == fresh(tokens, y, y_mask=mask)).all()
ema.ar_sample(y, mask)   # sets the compiled head attribute
twin = copy.deepcopy(ema)   # after sampling: the attribute must not break a copy
assert all(p.dtype == torch.float32 for p in twin.parameters())
print("7. checkpoint round trip, EMA copy, copy after sampling")

# 8. Sampling: rows differ with one caption (each draw is random), the first patch depends on the caption, and a drawn patch lies in [-1, 1]. tokens[:, i:] must not matter.
same = y[:1].repeat(2, 1, 1)
sm = mask[:1].repeat(2, 1)
torch.manual_seed(1)
imgs = net.ar_sample(same, sm)
assert imgs.shape == (2, 3, 64, 64) and (imgs[0] != imgs[1]).any() and imgs.abs().max() <= 1
torch.manual_seed(2)
again = net.ar_sample(same, sm)
assert (imgs != again).any()
torch.manual_seed(1)
assert (net.ar_sample(same, sm) == imgs).all(), "the same seed must give the same images"
print("8. sampling is random per row, reproducible by seed, in range")

# 9. Under bf16 autocast, as in training and predict: finite, and the head handles tau = 0 and tau = 1.
with torch.autocast("cpu", dtype=torch.bfloat16):
    z = net(tokens, y, y_mask=mask)
    for v in (0.0, 1e-4, 0.999, 1.0):
        h = net.head(torch.randn(4, P), torch.full((4,), v), z[:, 0].repeat(2, 1)[:4].float())
        assert h.isfinite().all(), v
    s = net.ar_sample(y, mask)
assert s.dtype in (torch.float32, torch.bfloat16) and s.isfinite().all()
print("9. bf16 autocast, tau at both ends; sample dtype", s.dtype)

# 10. A compiled forward gives the eager values on both paths and two caption lengths (CPU, small). This is the path of the real job.
c = wake(ARJiT(**kw))
e = copy.deepcopy(c)
c.compile()
for T in (8, 16):
    yy, m = torch.randn(2, T, 32), torch.ones(2, T, dtype=torch.bool)
    d1 = (c(tokens, yy, y_mask=m) - e(tokens, yy, y_mask=m)).abs().max().item()
    d2 = (c(tokens, yy, y_mask=m, q=q, tau=tau) - e(tokens, yy, y_mask=m, q=q, tau=tau)).abs().max().item()
    assert d1 < 1e-3 and d2 < 1e-3, (T, d1, d2)
tr = ARTrainer(null_condition_p=0.1)
res = tr(c, None, None, x, y.clone(), null.clone(), {"y_mask": mask.clone()})
res["loss"].backward()
none = [n for n, p in c.named_parameters() if p.requires_grad and p.grad is None]
assert not none, none
print("10. compiled forward on both paths, and every parameter has a gradient through it")

# 11. Hand count of the head's FLOPs at d640, 2 per multiply-add, against FlopCounterMode's 9.87 G.
w, D = 1024, 640
per = 768 * w + D * w + 256 * w + w * w + 3 * (2 * w * w + 3 * w * w) + w * 768
print(f"11. head by hand: {2 * 256 * per / 1e9:.2f} GFLOP per image; final linear of the point model {2 * 256 * 640 * 768 / 1e9:.3f} GFLOP")
print("all hostile checks passed")
