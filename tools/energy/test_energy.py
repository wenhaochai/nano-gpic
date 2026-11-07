#!/usr/bin/env python3
"""Hostile CPU checks of test E (docs/energy_head.md): JiT_T2I(head_width > 0), trained by ImplicitJiTTrainer and sampled by ImplicitSampler.
Run on a vis node from baselines/PixelGen with PYTHONPATH=. : python ../../tools/energy/test_energy.py"""
import subprocess
import sys
import types

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.sampling_implicit import ImplicitSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.training_implicit import ImplicitJiTTrainer
from src.diffusion.flow_matching.training_JiT import constant
from src.models.transformer.JiT_T2I import JiT_T2I

small = dict(input_size=64, in_channels=3, num_groups=4, hidden_size=64, num_blocks=3, num_text_blocks=1, patch_size=16, txt_embed_dim=32, txt_max_length=32, bottleneck_dim=16)
d640 = dict(input_size=256, in_channels=3, num_groups=10, hidden_size=640, num_blocks=9, num_text_blocks=2, patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256)
x = torch.randn(4, 3, 64, 64); t = torch.rand(4); y = torch.randn(4, 8, 32); mask = torch.ones(4, 8, dtype=torch.bool); mask[:, 6:] = False
trainer = lambda **kw: ImplicitJiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0, null_condition_p=0.0, samples=2, beta=1.0, **kw)


def head_flops(L=256, D=640, W=352, X=64, P=768, depth=2, m=2):
    """Matmul FLOPs (2 per MAC) of EnergyHead per input: the projections once, the noise layers, blocks and output m times."""
    once = L * D * W + D * W
    per = L * X * W + X * W + depth * (2 * L * W * W + 3 * W * W) + L * W * P
    return 2 * (once + m * per)


# 1. Off switch: head_width = 0 is the model of the last commit before this change, on one seed: the same parameters, the same RNG stream, the same outputs
src = subprocess.run(["git", "show", "f429d51:baselines/PixelGen/src/models/transformer/JiT_T2I.py"], capture_output=True, text=True, check=True).stdout
ref = types.ModuleType("JiT_T2I_ref"); exec(compile(src, "JiT_T2I_ref", "exec"), ref.__dict__)
torch.manual_seed(7); a = ref.JiT_T2I(**small); after_a = torch.rand(3)
torch.manual_seed(7); b = JiT_T2I(**small); after_b = torch.rand(3)
assert torch.equal(after_a, after_b), "the off model draws a different RNG stream at init"
sa, sb = a.state_dict(), b.state_dict()
assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
y32 = torch.randn(4, 32, 32)   # the official unmasked path pads y to txt_max_length
assert torch.equal(a(x, t, y32), b(x, t, y32)) and torch.equal(a(x, t, y, y_mask=mask), b(x, t, y, y_mask=mask))
for p in a.parameters():   # nonzero output layer, so the comparison is not of two zero images
    p.data.normal_(std=0.05)
b.load_state_dict(a.state_dict())
assert torch.equal(a(x, t, y, y_mask=mask), b(x, t, y, y_mask=mask)) and a(x, t, y, y_mask=mask).abs().max() > 1e-3

# 2. Shapes, and the head's parameters are the only new ones besides FinalLayer's
torch.manual_seed(1); e = JiT_T2I(**small, head_width=48)
assert e(x, t, y, y_mask=mask).shape == (4, 3, 64, 64) and e(x, t, y, y_mask=mask, m=2).shape == (8, 3, 64, 64)
new = {k for k in dict(e.named_parameters()) if k not in dict(b.named_parameters())}
assert all(k.startswith("final_layer.") for k in new) and "final_layer.out.linear.weight" in new, new

# 3. One backbone pass for both samples; the head runs on both; the samples of one input differ; a new call draws new noise
e.final_layer.out.linear.weight.data.normal_(std=0.05)
for blk in e.final_layer.blocks:
    blk.adaLN_modulation[-1].weight.data.normal_(std=0.05)
calls = []
for i, blk in enumerate(e.blocks):
    blk.register_forward_hook(lambda mod, inp, out, i=i: calls.append(("backbone", i, out.shape[0])))
for i, blk in enumerate(e.final_layer.blocks):
    blk.register_forward_hook(lambda mod, inp, out, i=i: calls.append(("head", i, out.shape[0])))
g = e(x, t, y, y_mask=mask, m=2)
assert calls == [("backbone", i, 4) for i in range(3)] + [("head", i, 8) for i in range(2)], calls
assert (g[0] - g[1]).abs().mean() > 1e-3 and (g[2] - g[3]).abs().mean() > 1e-3, "two samples of one input are equal"
assert (e(x, t, y, y_mask=mask) - e(x, t, y, y_mask=mask)).abs().mean() > 1e-3, "two calls give the same sample"

# 4. Gradient reaches the backbone and the head, from the official init, after a few AdamW steps (at init the zero output layer and adaLN gates stop it, as in the point model)
torch.manual_seed(2); e = JiT_T2I(**small, head_width=48)
tr, opt = trainer(), torch.optim.AdamW(e.parameters(), lr=1e-3)
xs = torch.randn(4, 3, 64, 64)
for _ in range(4):
    opt.zero_grad()
    tr._impl_trainstep(e, None, None, xs, y, {"y_mask": mask})["loss"].backward()
    opt.step()
need = ["x_embedder.proj1.weight", "blocks.0.attn.qkv_x.weight", "blocks.2.mlp.w12.weight", "text_refine_blocks.0.attn.qkv.weight", "final_layer.inp.weight", "final_layer.cond.weight",
        "final_layer.xi_token.weight", "final_layer.xi_global.weight", "final_layer.blocks.0.mlp.0.weight", "final_layer.blocks.1.adaLN_modulation.1.weight", "final_layer.out.linear.weight"]
grads = dict(e.named_parameters())
for k in need:
    assert k in grads, k
    assert grads[k].grad is not None and grads[k].grad.abs().max() > 0, f"no gradient at {k}"
with torch.no_grad():
    g = e(xs, t, y, y_mask=mask, m=2)
assert (g[0] - g[1]).abs().mean() > 1e-4, "the samples are still equal after training starts"

# 5. The loss on a hand example: x = 0, t = 0.5 (so w = 2), samples of constant value 1 and 3. to_data = (2 + 6) / 2 = 4, between = 2 * 2 = 4, score = 4 - 4 / 2 = 2
class Fixed(torch.nn.Module):
    def __init__(self, g): super().__init__(); self.g = g
    def forward(self, x_t, t, y, y_mask=None, m=1): return self.g
hand = trainer(P_mean=0.0, P_std=0.0)
gk = torch.cat([torch.full((1, 1, 2, 2), 1.0), torch.full((1, 1, 2, 2), 3.0)])
out = hand._impl_trainstep(Fixed(gk), None, None, torch.zeros(1, 1, 2, 2), y[:1], None)
assert abs(out["to_data"].item() - 4) < 1e-4 and abs(out["between"].item() - 4) < 1e-4 and abs(out["loss"].item() - 2) < 1e-4, out

# 6. Repulsion has the right sign: with x placed so that the pull to the data is orthogonal to g1 - g2, descent moves the two samples apart
g1 = torch.tensor([[[[1.0, 0.0]]], [[[-1.0, 0.0]]]], requires_grad=True)
hand._impl_trainstep(Fixed(g1), None, None, torch.tensor([[[[1.0, 1.0]]]]), y[:1], None)["loss"].backward()
assert g1.grad[0, 0, 0, 0] < 0 < g1.grad[1, 0, 0, 0], g1.grad   # sample 1 sits at +1 and must move up, sample 2 at -1 and must move down

# 7. FLOPs against the hand table: the d640 model with the head at m = 2 against the point model, the difference is the head minus FinalLayer's Linear
def fwd_flops(net, m):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(torch.randn(1, 3, 256, 256), torch.rand(1), torch.randn(1, 8, 2048), y_mask=torch.ones(1, 8, dtype=torch.bool), **({"m": m} if m > 1 else {}))
    return fc.get_total_flops()
final = 2 * 256 * 640 * 768
point = fwd_flops(JiT_T2I(**d640), 1)
energy = fwd_flops(JiT_T2I(**d640, head_width=352), 2)
assert energy - point == head_flops() - final, (energy - point, head_flops() - final)
assert abs(head_flops() - final - 0.0194 * 34.72e9) < 0.0005 * 34.72e9, head_flops() - final

# 8. Sampling is reproducible by seed, and another seed gives other samples
torch.manual_seed(3); e = JiT_T2I(**small, head_width=48)
for p in e.parameters():
    p.data.normal_(std=0.05)
sam = ImplicitSampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=8, guidance=1.0)
def draw(seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return sam._impl_sampling(lambda z, tt, c: e(z, tt, c, y_mask=mask), torch.randn(4, 3, 64, 64), y, None)[0][-1]
assert torch.equal(draw(11), draw(11)) and (draw(11) - draw(12)).abs().mean() > 1e-3
torch.manual_seed(11); noise = torch.randn(4, 3, 64, 64)
with torch.no_grad():   # fresh head noise at each call: the same start, two draws, two results
    assert (sam._impl_sampling(lambda z, tt, c: e(z, tt, c, y_mask=mask), noise, y, None)[0][-1] - sam._impl_sampling(lambda z, tt, c: e(z, tt, c, y_mask=mask), noise, y, None)[0][-1]).abs().mean() > 1e-3

print(f"all checks passed; d640 forward GFLOP (CPU counter): point {point / 1e9:.3f}, energy m=2 {energy / 1e9:.3f}, head {head_flops() / 1e9:.4f}, added {(energy - point) / 1e9:.4f} = {(energy - point) / 34.72e9:.2%} of 34.72")
