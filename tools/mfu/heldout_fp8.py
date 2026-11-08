#!/usr/bin/env python3
"""Held-out loss of one checkpoint's raw weights, evaluated the way it was trained (gpic-28 2026-11-08, owner: 「评测用的是 bf16 主权重
是为啥，不应该train什么eval什么吗」). The d16 FP8 check lost FD +9.9% at C0 when both arms were scored in bf16; is the FP8-trained model
bad, or only bad when run in bf16? Variants, all with tools/heldout_loss.py's data, levels, noise and loss (3072 val images x 16 levels,
seed 0, the loss in fp32 under bf16 autocast):
  raw    the tool's own evaluation (fp32 master weights, bf16 autocast): must repeat the heldout_loss.jsonl line
  bf16w  the master weights rounded to bf16 first, as the training's FSDP2 forward saw them
  fp8    bf16w, then the image blocks' 64 GEMMs in FP8 as in training (src/utils/fp8.py native2 = native3's forward with each
         tensor's current amax in place of the 16-step delayed scale)
The model code is the FP8 check's own (gpic-wt/sweep_fp8_cap_run, its MFU switches off). Appends one JSON line to <out>.
usage: heldout_fp8.py <ckpt> <raw|bf16w|fp8> <out.jsonl>"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/gpic-wt/heldout_loss_run/tools")
import heldout_loss as H  # noqa: E402
CODE = "/scratch/gpfs/GROUP/USER/project/gpic-wt/sweep_fp8_cap_run"
sys.path[:] = [p for p in sys.path if not p.startswith(H.REPO)]
sys.path.insert(0, os.path.join(CODE, "baselines", "PixelGen"))
ckpt, variant, out = sys.argv[1], sys.argv[2], sys.argv[3]
assert variant in ("raw", "bf16w", "fp8"), variant
t0 = time.time()
dev = "cuda"
torch.set_num_threads(2)
data = H.load_data(H.data_path(3072, 3072), dev)
kw, loss = H.run_config(os.path.dirname(ckpt))
ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
step = ck["global_step"]
net = H.build_denoiser(kw, ck["state_dict"], "denoiser.", dev)
del ck
if variant in ("bf16w", "fp8"):
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(p.to(torch.bfloat16).float())
if variant == "fp8":
    from src.utils.fp8 import convert_fp8
    convert_fp8(net, "native2")
tr = H.make_trainer(loss)
ts = H.levels(loss["P_mean"], loss["P_std"], loss["timeshift"])
res = H.evaluate({variant: net}, tr, data["val"], ts, H.SEED, H.BATCH, dev, True, 4)[variant]
lm, im = res["level_mean"], res["image_mean"]
rec = dict(ckpt=ckpt, step=step, variant=variant, loss_val=float(lm.mean()), loss_val_t=lm.tolist(),
           loss_val_se=float(im.std() / math.sqrt(len(im))), t=ts.tolist(), gpu=torch.cuda.get_device_name(dev),
           job=os.environ.get("FETCHED_JOB_ID") or os.environ.get("SLURM_JOB_ID", ""), elapsed_s=round(time.time() - t0, 1))
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"HELDOUT_FP8 {variant} step {step} {os.path.basename(os.path.dirname(os.path.dirname(ckpt)))}: val {rec['loss_val']:.5f}; "
      f"{rec['elapsed_s']:.0f} s", flush=True)
