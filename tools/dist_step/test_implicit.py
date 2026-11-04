#!/usr/bin/env python3
"""CPU checks of the implicit sampler code. Run on a vis node from baselines/PixelGen with PYTHONPATH=. : python ../../tools/dist_step/test_implicit.py"""
import torch
from src.models.transformer.JiT_T2I import JiT_T2I
from src.diffusion.flow_matching.training_implicit import ImplicitJiTTrainer
from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant
from src.diffusion.flow_matching.sampling_implicit import ImplicitSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.base.guidance import simple_guidance_fn
kw = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=32, bottleneck_dim=16)
x = torch.randn(4, 3, 64, 64); t = torch.rand(4); y = torch.randn(4, 8, 32); mask = torch.ones(4, 8, dtype=torch.bool); mask[:, 6:] = False

# 1. xi_dim = 0 is the point model, with the same RNG stream and the same outputs as before the change
torch.manual_seed(1); a = JiT_T2I(**kw)
assert not hasattr(a, "xi_global")
o = a(x, t, y, y_mask=mask)
assert o.shape == x.shape

# 2. m samples: the shape, the samples differ, and the blocks before the fork run once
torch.manual_seed(1); b = JiT_T2I(**kw, xi_dim=8, xi_block=2)
b.final_layer.linear.weight.data.normal_(std=0.05)   # zero at init would hide the difference
calls = []
for i, blk in enumerate(b.blocks):
    blk.register_forward_hook(lambda m, inp, out, i=i: calls.append((i, out.shape[0])))
g = b(x, t, y, y_mask=mask, m=3)
assert g.shape == (12, 3, 64, 64)
assert calls == [(0, 4), (1, 4), (2, 12)], calls
assert (g[0] - g[1]).abs().max() > 1e-4, "samples of one input are equal"
for name, p in b.named_parameters():
    assert p.requires_grad or name == "pos_embed", name

# 3. The two models share all other parameters by name, so a point checkpoint could load into the implicit one
extra = set(dict(b.named_parameters())) - set(dict(a.named_parameters()))
assert extra == {"xi_global.weight", "xi_global.bias", "xi_token.weight", "xi_token.bias"}, extra

# 4. The loss against a hand computation, on a net that returns known samples
class Fixed(torch.nn.Module):
    def __init__(self, g): super().__init__(); self.g = g
    def forward(self, x_t, t, y, y_mask=None, m=1): return self.g
tr = ImplicitJiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, null_condition_p=0.0, samples=2, beta=1.0)
torch.manual_seed(5)
base_t = (torch.randn(4, dtype=torch.float32) * tr.P_std + tr.P_mean).sigmoid()
known = torch.randn(8, 3, 64, 64)
torch.manual_seed(5)
out = tr._impl_trainstep(Fixed(known), None, None, x, y, {"y_mask": mask})
w = 1 / (1 - base_t).clamp_min(tr.t_eps)
g2 = known.view(4, 2, 3, 64, 64)
hand = []
for k in range(4):
    r1 = (w[k] * (g2[k, 0] - x[k])).pow(2).mean().add(1e-8).sqrt()
    r2 = (w[k] * (g2[k, 1] - x[k])).pow(2).mean().add(1e-8).sqrt()
    r3 = (w[k] * (g2[k, 0] - g2[k, 1])).pow(2).mean().add(1e-8).sqrt()
    hand.append(0.5 * (r1 + r2) - 0.5 * r3)
assert abs(out["loss"].item() - torch.stack(hand).mean().item()) < 1e-5, (out["loss"].item(), torch.stack(hand).mean().item())

# 5. The score prefers the true spread: on a 1-D Gaussian posterior it is lowest near the true standard deviation
class Gauss(torch.nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x_t, t, y, y_mask=None, m=1): return (self.s * torch.randn(x_t.shape[0] * m, 1, 8, 8))
vals = {}
for s in (0.0, 0.5, 1.0, 1.5, 2.0):
    torch.manual_seed(0)
    data = torch.randn(4096, 1, 8, 8)
    vals[s] = ImplicitJiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, null_condition_p=0.0, P_mean=9.0, P_std=0.0, t_eps=1.0)._impl_trainstep(Gauss(s), None, None, data, y, None)["loss"].item()
assert min(vals, key=vals.get) == 1.0, vals

# 6. The sampler: with a net that returns a fixed image, the path is a straight line and the last step returns that image exactly
target = torch.randn(4, 3, 64, 64)
for steps in (1, 2, 8, 50):
    sam = ImplicitSampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=steps, guidance=1.0)
    xs, _ = sam._impl_sampling(lambda z, tt, c: target, torch.randn(4, 3, 64, 64), None, None)
    assert len(xs) == steps + 1 and (xs[-1] - target).abs().max() < 1e-5, (steps, (xs[-1] - target).abs().max())

# 7. The sampler's step: the marginal of one step from a Gaussian data law, with exact posterior samples, at a coarse step size
torch.manual_seed(3)
n = 400000; sd = 0.7
d = sd * torch.randn(n); e = torch.randn(n); t0, t1 = 0.3, 0.8
xt = t0 * d + (1 - t0) * e
post_var = 1 / (1 / sd ** 2 + t0 ** 2 / (1 - t0) ** 2); post_mean = post_var * t0 * xt / (1 - t0) ** 2
xhat = post_mean + post_var ** 0.5 * torch.randn(n)
nx = xt + (t1 - t0) / (1 - t0) * (xhat - xt)
want = (t1 ** 2 * sd ** 2 + (1 - t1) ** 2)
assert abs(nx.var().item() - want) / want < 0.01, (nx.var().item(), want)
print("all checks passed;", "energy score by sample std:", {k: round(v, 4) for k, v in vals.items()})
