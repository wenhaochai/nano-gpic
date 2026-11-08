#!/usr/bin/env python3
"""Where does FP8 lose precision in the image blocks? (owner 2026-11-07, after the d17 check failed: Megatron and DeepSeek
keep parts in high precision; which parts matter here?) On one ladder checkpoint (raw weights), B held-out val images at
a few noise levels, the training loss forward and backward in bf16 autocast, as tools/heldout_loss.py computes it. At each
GEMM that native3 runs in FP8 (qkv_x, proj, w12 and w3 of every image block, forward; the w3 and w12 dgrads, from dy and
dz) the relative error ||Y_q - Y|| / ||Y|| of the GEMM output when both inputs are rounded to FP8 (fp32 products and sums;
Y from the same bf16 inputs), under four schemes:
  T4  one e4m3 scale per tensor (native3; with the current amax, which delayed scaling's history max can only coarsen),
  T5  as T4 with the gradient in e5m2 (Transformer Engine's default HYBRID format; dgrads only),
  R4  e4m3 per row: a scale per token row of the activation or gradient, per output row (forward) or input column
      (dgrad) of the weight (torchao's rowwise),
  B4  e4m3 per 1 x 128 block of the activation or gradient and 128 x 128 block of the weight (DeepSeek-V3).
Also each input's crest factor amax / rms. Forward fast accumulation is not modelled. A100 is enough (casts only).
usage: python fp8_site_error.py <ms-step ckpt> [<images> [<levels, comma-separated indices of the 16>]]"""
import json
import os
import sys

import torch

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/heldout_loss_run/tools")
import heldout_loss as H  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
FMAX = {E4: 448.0, E5: 57344.0}
ckpt = os.path.abspath(sys.argv[1])
B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
LV = [int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 5, 10, 14]
dev = "cuda"


def q(x, fmt, gr, gc):
    """x [R, C] fp32 rounded to fmt with one scale per gr x gc block (0 = the whole dimension), dequantized."""
    R, C = x.shape
    gr, gc = gr or R, gc or C
    Rp, Cp = -(-R // gr) * gr, -(-C // gc) * gc
    b = torch.nn.functional.pad(x, (0, Cp - C, 0, Rp - R)).view(Rp // gr, gr, Cp // gc, gc)
    s = FMAX[fmt] / b.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    return ((b * s).clamp(-FMAX[fmt], FMAX[fmt]).to(fmt).float() / s).view(Rp, Cp)[:R, :C]


FWD = {"T4": ((E4, 0, 0), (E4, 0, 0)), "R4": ((E4, 1, 0), (E4, 1, 0)), "B4": ((E4, 1, 128), (E4, 128, 128))}
BWD = {"T4": ((E4, 0, 0), (E4, 0, 0)), "T5": ((E5, 0, 0), (E4, 0, 0)), "R4": ((E4, 1, 0), (E4, 0, 1)),
       "B4": ((E4, 1, 128), (E4, 128, 128))}
stats = {}   # (site, block, scheme) -> [errors over levels]; (site, block, "crest") -> [crest factors]


def add(key, v):
    stats.setdefault(key, []).append(float(v))


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def fwd_hook(site, blk):
    @torch.autocast("cuda", enabled=False)   # the hooks run inside the loss's bf16 autocast: keep their GEMMs in fp32
    def hook(mod, inp):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.bfloat16).float()
        w = mod.weight.detach().to(torch.bfloat16).float()
        y = x @ w.t()
        add((site, blk, "crest"), x.abs().max() / x.pow(2).mean().sqrt())
        for name, ((fx, xr, xc), (fw, wr, wc)) in FWD.items():
            add((site, blk, name), rel(q(x, fx, xr, xc) @ q(w, fw, wr, wc).t(), y))
    return hook


def bwd_hook(site, blk):
    def fwd(mod, inp, out):
        w = mod.weight

        @torch.autocast("cuda", enabled=False)
        def on_grad(g):
            g2 = g.detach().reshape(-1, g.shape[-1]).to(torch.bfloat16).float()
            w2 = w.detach().to(torch.bfloat16).float()
            dx = g2 @ w2
            add((site, blk, "crest"), g2.abs().max() / g2.pow(2).mean().sqrt())
            for name, ((fg, gr, gc), (fw, wr, wc)) in BWD.items():
                add((site, blk, name), rel(q(g2, fg, gr, gc) @ q(w2, fw, wr, wc), dx))
        out.register_hook(on_grad)
    return fwd


spec = H.spec_of(ckpt)
kw, loss_cfg = H.run_config(os.path.dirname(ckpt))
sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
net = H.build_denoiser(kw, sd, "denoiser.", dev)
del sd
mods = dict(net.named_modules())
nblk = len(net.blocks)
for i in range(nblk):
    for site, name in (("qkv_x", f"blocks.{i}.attn.qkv_x"), ("proj", f"blocks.{i}.attn.proj"),
                       ("w12", f"blocks.{i}.mlp.w12"), ("w3", f"blocks.{i}.mlp.w3")):
        assert isinstance(mods.get(name), torch.nn.Linear), name
        mods[name].register_forward_pre_hook(fwd_hook(site, i))
    mods[f"blocks.{i}.mlp.w3"].register_forward_hook(bwd_hook("dgrad_w3(dy)", i))
    mods[f"blocks.{i}.mlp.w12"].register_forward_hook(bwd_hook("dgrad_w12(dz)", i))
tr = H.make_trainer(loss_cfg)
data = H.load_data(H.data_path(3072, 3072), "cpu")
split = data["val"]
idx = split["order"][::len(split["order"]) // B][:B]   # spread over caption lengths
x = H.to_input(split["img"][torch.tensor(idx)]).to(dev)
y, mask = H.text_batch(split, idx, dev)
ts = H.levels(loss_cfg["P_mean"], loss_cfg["P_std"], loss_cfg["timeshift"])
shape = tuple(x.shape[1:])
noise = torch.stack([H.image_noise(H.SEED, j, len(ts), shape) for j in idx]).to(dev)   # [B, num_t, 3, H, W]
for k in LV:
    net.zero_grad(set_to_none=True)
    H.level_loss(net, tr, x, y, mask, ts[k].to(dev), noise[:, k], True).mean().backward()
sites = ["qkv_x", "proj", "w12", "w3", "dgrad_w3(dy)", "dgrad_w12(dz)"]
med = lambda v: sorted(v)[len(v) // 2]
print(f"{spec['size']} step {spec['step']} ({ckpt}), raw weights, {B} val images, levels {LV} of {len(ts)}, "
      f"{nblk} image blocks: relative error of the GEMM output, mean over levels; median / max over blocks (block)")
out = {"ckpt": ckpt, "images": B, "levels": LV, "sites": {}}
for site in sites:
    schemes = list(BWD) if site.startswith("dgrad") else list(FWD)
    per = {sc: [sum(stats[(site, i, sc)]) / len(stats[(site, i, sc)]) for i in range(nblk)] for sc in schemes + ["crest"]}
    out["sites"][site] = per
    cells = []
    for sc in schemes:
        v = per[sc]
        i = max(range(nblk), key=lambda j: v[j])
        cells.append(f"{sc} {100 * med(v):5.2f}% / {100 * v[i]:5.2f}% (b{i})")
    print(f"  {site:14s} " + "  ".join(cells) + f"  crest {med(per['crest']):6.1f} / {max(per['crest']):6.1f}")
print("T4 per block (%): " + "; ".join(f"{s}: " + " ".join(f"{100 * e:.2f}" for e in out["sites"][s]["T4"]) for s in sites))
dst = os.path.join("/scratch/gpfs/GROUP/USER/runs/gpic-speed/mfu", f"fp8_site_error_{spec['size']}_{spec['step']}.json")
json.dump(out, open(dst, "w"))
print("wrote", dst)
print("FP8_SITE_ERROR DONE")
