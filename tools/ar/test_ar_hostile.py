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
before = torch.get_rng_state()
torch.manual_seed(5)
one = net._read(torch.ones(2, 3))
assert (one == 1).all() and (torch.manual_seed(5), torch.randn(1))[1] == (torch.manual_seed(5), torch.randn(1))[1]
torch.set_rng_state(before)
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

# 12. context_noise. At 0 the trainer gives the first design's loss on the same RNG stream. Above 0 the backbone reads noisy patches, the target stays clean, causality holds, and sampling returns clean drawn patches.
plain, noisy = wake(ARJiT(**kw)), None
tr0 = ARTrainer(null_condition_p=0.0)
torch.manual_seed(9)
l0 = tr0._impl_trainstep(plain, None, None, x, y, {"y_mask": mask})["loss"].item()
torch.manual_seed(9)
l1 = tr0._impl_trainstep(plain, None, None, x, y, {"y_mask": mask})["loss"].item()
assert l0 == l1
noisy = wake(ARJiT(**kw, context_noise=0.3))
seen2 = {}
orig = ARJiT.forward
def spy_forward(self, tokens, y, y_mask=None, q=None, tau=None):
    seen2["tokens"] = tokens.clone()
    return orig(self, tokens, y, y_mask=y_mask, q=q, tau=tau)
ARJiT.forward = spy_forward
tr0._impl_trainstep(noisy, None, None, x, y, {"y_mask": mask})
clean = torch.nn.functional.unfold(x, kernel_size=16, stride=16).transpose(1, 2)
gap = (seen2["tokens"] - clean)
assert abs(gap.std().item() - 0.3) < 0.02, gap.std().item()
tr0._impl_trainstep(plain, None, None, x, y, {"y_mask": mask})
assert (seen2["tokens"] == clean).all(), "context_noise = 0 must read the clean patches"
torch.manual_seed(3)
img = noisy.ar_sample(y, mask)
read = seen2["tokens"]   # the last forward's context: all but the last patch are filled
out = torch.nn.functional.unfold(img, kernel_size=16, stride=16).transpose(1, 2)
assert img.abs().max() <= 1, "returned patches are the clean draws"
assert abs((read[:, :15] - out[:, :15]).std().item() - 0.3) < 0.03
ARJiT.forward = orig
print("12. context_noise: the backbone reads noise, the target and the output stay clean")

# 13. prefix: the first k patches come back unchanged, the draw depends on them, and k = 0 and None agree on one seed.
pre = torch.rand(2, 5, P) * 2 - 1
img = net.ar_sample(y, mask, prefix=pre)
back = torch.nn.functional.unfold(img, kernel_size=16, stride=16).transpose(1, 2)
assert (back[:, :5] - pre).abs().max() < 1e-5
torch.manual_seed(4)
a0 = net.ar_sample(y, mask)
torch.manual_seed(4)
a1 = net.ar_sample(y, mask, prefix=torch.zeros(2, 0, P))
assert (a0 == a1).all()
torch.manual_seed(4)
a2 = net.ar_sample(y, mask, prefix=pre)
assert (a2 != a0).any()
print("13. prefix: returned unchanged, and it changes the rest")

# 14. Random order: no leak from unknown tokens, known tokens do reach every output, the loss counts masked tokens only, every parameter trains, and the sampler draws each token exactly once.
rnd = wake(ARJiT(**kw, order="random", ar_steps=6))
known = torch.zeros(2, N, dtype=torch.bool)
known[0, [1, 5]] = True
known[1, [2, 9]] = True
t = tokens.clone().requires_grad_(True)
z = rnd(t, y, y_mask=mask, known=known)
g = torch.autograd.grad(z.sum(), t)[0].abs().amax(dim=2)
assert (g[~known] == 0).all() and (g[known] > 0).all(), g
assert (rnd(tokens, y, y_mask=mask, known=torch.ones(2, N, dtype=torch.bool)) != rnd(tokens, y, y_mask=mask, known=known)).any()
tr_r = ARTrainer(null_condition_p=0.0, min_mask=0.7)
torch.manual_seed(11)
rnd.zero_grad()
for _ in range(8):   # a step with every token masked gives the patch embedding a zero gradient, as MAR does; it stays in the graph, so DDP is fine
    res = tr_r._impl_trainstep(rnd, None, None, x, y, {"y_mask": mask})
    res["loss"].backward()
    assert all(p.grad is not None for p in rnd.parameters() if p.requires_grad)
none = [n for n, p in rnd.named_parameters() if p.requires_grad and p.grad.abs().sum() == 0]
assert not none and res["loss_per_sample"].shape == (2,), none
assert not hasattr(rnd, "causal") and not hasattr(rnd, "bos") and not hasattr(net, "mask_token")
rates = []
class Rec(ARJiT):
    def forward(self, tokens, y, y_mask=None, q=None, tau=None, known=None):
        rates.append(1 - known.float().mean().item())
        assert (known.sum(1) == known[0].sum()).all()
        return super().forward(tokens, y, y_mask=y_mask, q=q, tau=tau, known=known)
rec = Rec(**kw, order="random", ar_steps=6)
for _ in range(40):
    tr_r._impl_trainstep(rec, None, None, x, y, {"y_mask": mask})
assert min(rates) >= 0.7 and max(rates) <= 1.0 and sum(rates) / len(rates) > 0.8, (min(rates), max(rates))
drawn_at = []
orig_draw = ARJiT.draw_patch
def spy_draw(self, z):
    drawn_at.append(z.shape[0])
    return orig_draw(self, z)
ARJiT.draw_patch = spy_draw
img = rnd.ar_sample(y, mask)
assert img.shape == (2, 3, 64, 64) and sum(drawn_at) == 2 * N and all(c % 2 == 0 for c in drawn_at), drawn_at
assert len(drawn_at) <= 6
drawn_at.clear()
big_steps = wake(ARJiT(**kw, order="random", ar_steps=64))
big_steps.ar_sample(y, mask)
assert sum(drawn_at) == 2 * N and len(drawn_at) == N, (sum(drawn_at), len(drawn_at))   # more steps than tokens: one token per step
drawn_at.clear()
pre = torch.rand(2, 5, P) * 2 - 1
img = rnd.ar_sample(y, mask, prefix=pre)
back = torch.nn.functional.unfold(img, kernel_size=16, stride=16).transpose(1, 2)
assert (back[:, :5] - pre).abs().max() < 1e-5 and sum(drawn_at) == 2 * (N - 5), drawn_at
ARJiT.draw_patch = orig_draw
c2 = wake(ARJiT(**kw, order="random", ar_steps=6))
e2 = copy.deepcopy(c2)
c2.compile()
assert (c2(tokens, y, y_mask=mask, known=known) - e2(tokens, y, y_mask=mask, known=known)).abs().max() < 1e-3
res = tr_r(c2, None, None, x, y.clone(), null.clone(), {"y_mask": mask.clone()})
res["loss"].backward()
assert not [n for n, p in c2.named_parameters() if p.requires_grad and p.grad is None]
print("14. random order: no leak, loss on masked tokens, each token drawn once, compiled path")

# 19. HeadFlow is the same flow as the loop: same seed, same draws.
hf = wake(ARJiT(**kw, order="random", ar_steps=64))
zz = torch.randn(7, 64)
torch.manual_seed(11)
d_loop = hf.draw_patch(zz)
os.environ["GPIC_AR_HEAD_FLOW"] = "1"
torch.manual_seed(11)
d_flow = hf.draw_patch(zz)
hf.head_steps += 2     # the flow follows a change of steps (overfit_ar.py's check at 128)
torch.manual_seed(11)
d_flow4 = hf.draw_patch(zz)
del os.environ["GPIC_AR_HEAD_FLOW"]
torch.manual_seed(11)
d_loop4 = hf.draw_patch(zz)
assert (d_loop == d_flow).all() and (d_loop4 == d_flow4).all() and (d_loop != d_loop4).any()
print("19. HeadFlow draws equal the loop's, also after a change of steps")

# 18. The sampler variants through ar_sample, and the share of batches at few known tokens.
torch.manual_seed(8)
v_first = wake(ARJiT(**kw, order="random", ar_steps=64, redraw_first=8))
calls_first = []
keep_redraw = ARJiT.redraw
def spy_redraw(self, tokens, pick, y, y_mask=None):
    calls_first.append(pick.sum(1).tolist())
    return keep_redraw(self, tokens, pick, y, y_mask)
ARJiT.redraw = spy_redraw
img_first = v_first.ar_sample(y, mask)
ARJiT.redraw = keep_redraw
assert calls_first == [[8, 8]] and img_first.shape == (2, 3, 64, 64)
v_eta = wake(ARJiT(**kw, order="random", ar_steps=64, eta=2.0))
assert v_eta.ar_sample(y, mask).shape == (2, 3, 64, 64)
few = ARTrainer(null_condition_p=0.0, few_known_p=0.5, few_known_max=1)
seen_known = []
keep_fw = ARJiT.forward
def spy_known2(self, tokens, y, y_mask=None, q=None, tau=None, known=None):
    seen_known.append(known.sum(1).tolist())
    return keep_fw(self, tokens, y, y_mask=y_mask, q=q, tau=tau, known=known)
ARJiT.forward = spy_known2
tiny = wake(ARJiT(**kw, order="random", ar_steps=6))
xx = torch.rand(2, 3, 64, 64) * 2 - 1
for _ in range(120):
    few._impl_trainstep(tiny, None, None, xx, y, {"y_mask": mask})
ARJiT.forward = keep_fw
assert all(a == b for a, b in seen_known), "one known count per batch"
share_few = sum(a <= 1 for a, _ in seen_known) / len(seen_known)
assert 0.6 < share_few < 0.92, share_few   # on 16 tokens the MAR rule alone gives 0.50 at 1 known or fewer, so 0.5 + 0.5 x 0.50 = 0.75
print("18. ar_sample variants; few_known: share of batches at 1 known or fewer", round(share_few, 2))

# 17. sample_eta: at eta = 0 the schedule of known counts equals _sample_random's, every token gets drawn once and the head draws N per image. At eta > 0 the counts after each step still follow the schedule, tokens go back and are drawn again, nothing stays masked at the end, and the backbone calls stay ar_steps.
re = wake(ARJiT(**kw, order="random", ar_steps=64))
log_known, log_pick = [], []
keep_forward = ARJiT.forward
def spy_eta(self, tokens, y, y_mask=None, q=None, tau=None, known=None):
    log_known.append(known.clone())
    return keep_forward(self, tokens, y, y_mask=y_mask, q=q, tau=tau, known=known)
ARJiT.forward = spy_eta
log_known.clear()
re._sample_random(y, mask)
base_counts = [k.sum(1).tolist() for k in log_known]
log_known.clear()
img0, draws0 = re.sample_eta(y, mask, eta=0.0)
assert [k.sum(1).tolist() for k in log_known] == base_counts, "eta = 0 changed the schedule"
assert draws0 == N and img0.shape == (2, 3, 64, 64)
log_known.clear()
torch.manual_seed(5)
img5, draws5 = re.sample_eta(y, mask, eta=20.0)
counts5 = [k.sum(1) for k in log_known]
assert len(log_known) <= 64 and draws5 > 2 * N, draws5
went_back = any(((a & ~b).sum() > 0) for a, b in zip(log_known[:-1], log_known[1:]))
assert went_back, "no token went back to the mask state at eta = 20"
assert img5.shape == (2, 3, 64, 64) and torch.isfinite(img5).all()
ARJiT.forward = keep_forward
print("17. sample_eta: eta = 0 keeps the schedule and draws each token once; eta = 20 takes tokens back,", round(draws5 / N, 1), "x the draws")

# 16. The order the sampler returns is the draw order: the last entries are drawn first. redraw changes only the picked tokens, and depends on the others.
rd = wake(ARJiT(**kw, order="random", ar_steps=64))
seen3 = []
orig_draw2 = ARJiT.draw_patch
real_forward = ARJiT.forward
def spy_known(self, tokens, y, y_mask=None, q=None, tau=None, known=None):
    seen3.append(known.clone())
    return real_forward(self, tokens, y, y_mask=y_mask, q=q, tau=tau, known=known)
ARJiT.forward = spy_known
img, order = rd.ar_sample(y, mask), None
seen3.clear()
img, order = rd._sample_random(y, mask, return_order=True)
ARJiT.forward = real_forward
assert img.shape == (2, 3, 64, 64) and order.shape == (2, N)
for step, kn in enumerate(seen3):   # at step s, s tokens are known (one per step here: more steps than tokens), and they are the last s in the order
    for b in range(2):
        want = set(order[b, N - int(kn[b].sum()):].tolist())
        assert want == set(kn[b].nonzero().flatten().tolist()), (step, b)
assert int(seen3[0].sum()) == 0 and int(seen3[-1][0].sum()) == N - 1
toks = torch.rand(2, N, P) * 2 - 1
pick = torch.zeros(2, N, dtype=torch.bool)
pick[0, [1, 7, 9]] = True
pick[1, [0, 2, 3]] = True
out = rd.redraw(toks, pick, y, mask)
assert (out[~pick] == toks[~pick]).all() and (out[pick] != toks[pick]).any()
torch.manual_seed(1)
o1 = rd.redraw(toks, pick, y, mask)
other = toks.clone()
other[~pick] = -other[~pick]
torch.manual_seed(1)
o2 = rd.redraw(other, pick, y, mask)
assert (o1[pick] != o2[pick]).any()
torch.manual_seed(1)
o3 = rd.redraw(toks + 0.5 * pick[..., None].float(), pick, y, mask)   # the old values of the picked tokens must not matter
assert (o3[pick] == o1[pick]).all()
print("16. the draw order and redraw: only picked tokens change, the old picked values do not matter")

# 15. temperature: 1 changes nothing on one seed; below 1 the draws are closer together on a fresh head that has learned nothing but a mean.
tt = wake(ARJiT(**kw, temperature=1.0))
torch.manual_seed(3)
a = tt.draw_patch(torch.zeros(64, 64))
torch.manual_seed(3)
base_t = torch.randn(64, 768)   # the same start noise
tt2 = copy.deepcopy(tt)
tt2.temperature = 0.5
torch.manual_seed(3)
b = tt2.draw_patch(torch.zeros(64, 64))
assert (a != b).any() and b.std(0).mean() < a.std(0).mean(), (a.std(0).mean(), b.std(0).mean())
print("15. temperature scales the start noise")

# 11. Hand count of the head's FLOPs at d640, 2 per multiply-add, against FlopCounterMode's 9.87 G.
w, D = 1024, 640
per = 768 * w + D * w + 256 * w + w * w + 3 * (2 * w * w + 3 * w * w) + w * 768
print(f"11. head by hand: {2 * 256 * per / 1e9:.2f} GFLOP per image; final linear of the point model {2 * 256 * 640 * 768 / 1e9:.3f} GFLOP")
print("all hostile checks passed")
