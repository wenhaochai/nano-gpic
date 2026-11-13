"""GPU check of ARTM's KV-cached sampling on a trained checkpoint (docs/artm_ladder.md; the smoke's stage K): teacher-forced artm_sample gives, at
every transition and patch, the head tokens z of the full masked forward on the same inputs.
  fp32 (no autocast, TF32 off): the gate. A cache bug shows here; exact up to float rounding, so max |z_full - z_cached| / rms(z) must be < 1e-3.
  bf16 autocast (GPIC_MFU_FUSED_NORM as the environment sets it): reported against bf16 rounding alone (the full forward in bf16 against fp32).
The first version gated bf16 at 0.05 and failed in smoke w2642 (0.11); on CPU with that checkpoint the fp32 gap was under 5e-5 and bf16 rounding alone
gave a max of 0.125 to 0.138 (tmp/ladder_prof/dist_phase2/kv_floor.py, 2026-11-13): the max over ~10^6 values is a tail statistic of rounding.
usage: python tools/artm_ladder/kv_check.py <ckpt> [captions per test, default 4]"""
import glob
import os
import sys

import torch
import yaml

from src.models.transformer.ARTM_JiT import ARTMJiT

ckpt = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
run = os.path.dirname(ckpt)
cfg = yaml.safe_load(open(sorted(glob.glob(os.path.join(run, "config-TrainerFn.FITTING-*.yaml")))[-1]))
den = cfg["model"]["denoiser"]
assert den["class_path"].endswith("ARTM_JiT.ARTMJiT"), den["class_path"]
kw = {k: v for k, v in den["init_args"].items() if k not in ("weight_path", "load_ema")}
sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
net = ARTMJiT(**kw)
net.load_state_dict({k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}, strict=True)
net = net.cuda().eval()
torch.manual_seed(0)
C, H = net.in_channels, kw.get("input_size", 256)
side = H // net.patch_size
N, P = side * side, C * net.patch_size ** 2
x0 = torch.randn(B, C, H, H, device="cuda")
T = 24
y = torch.randn(B, T, kw["txt_embed_dim"], device="cuda")
ym = torch.zeros(B, T, dtype=torch.bool, device="cuda")
for b in range(B):
    ym[b, :8 + 4 * b] = True
teach = [torch.randn(B, N, P, device="cuda") * 0.5 for _ in range(net.art_T)]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def run(bf16):
    ctxm = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    with torch.no_grad(), ctxm:
        _, zs = net.artm_sample(x0, y, ym, teacher=teach)
        full, xk = [], x0
        for k in range(net.art_T):
            ctx = torch.nn.functional.unfold(xk, kernel_size=net.patch_size, stride=net.patch_size).transpose(1, 2)
            full.append(net(ctx, torch.full((B,), k / net.art_T, device="cuda"), y, y_mask=ym, art_tgt=teach[k]).float())
            xk = net.unpatchify(teach[k], H, H)
    return [z.float() for z in zs], full


def gap(a, b, ref):
    return (a - b).abs().max().item() / ref.pow(2).mean().sqrt().item()


c32, f32 = run(False)
c16, f16 = run(True)
worst32 = 0.0
for k in range(net.art_T):
    g32, g16, fl = gap(f32[k], c32[k], f32[k]), gap(f16[k], c16[k], f16[k]), gap(f16[k], f32[k], f32[k])
    print(f"transition {k}: max |z_full - z_cached| / rms(z): fp32 {g32:.2e}; bf16 {g16:.4f} (bf16 rounding alone {fl:.4f})", flush=True)
    worst32 = max(worst32, g32)
ok = worst32 < 1e-3
print(f"kv check: fp32 worst {worst32:.2e} (fused norm {os.environ.get('GPIC_MFU_FUSED_NORM', '0')}); {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
