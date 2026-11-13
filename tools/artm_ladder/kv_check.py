"""GPU check of ARTM's KV-cached sampling on a trained checkpoint (docs/artm_ladder.md; the smoke's stage K): with the levers of training and evaluation
(bf16 autocast; GPIC_MFU_FUSED_NORM as the environment sets it), teacher-forced artm_sample gives, at every transition and patch, the head tokens z of the
full masked forward on the same inputs. Prints the largest difference relative to the tokens' rms; exit 1 above 0.05 (bf16 rounding is about 0.01).
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
worst = 0.0
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    img, zs = net.artm_sample(x0, y, ym, teacher=teach)
    xk = x0
    for k in range(net.art_T):
        ctx = torch.nn.functional.unfold(xk, kernel_size=net.patch_size, stride=net.patch_size).transpose(1, 2)
        zf = net(ctx, torch.full((B,), k / net.art_T, device="cuda"), y, y_mask=ym, art_tgt=teach[k]).float()
        d = (zf - zs[k].float()).abs().max().item() / zf.pow(2).mean().sqrt().item()
        print(f"transition {k}: max |z_full - z_cached| / rms(z) = {d:.4f}", flush=True)
        worst = max(worst, d)
        xk = net.unpatchify(teach[k], H, H)
print(f"kv check: worst {worst:.4f} (fused norm {os.environ.get('GPIC_MFU_FUSED_NORM', '0')}); {'PASS' if worst < 0.05 else 'FAIL'}")
sys.exit(0 if worst < 0.05 else 1)
