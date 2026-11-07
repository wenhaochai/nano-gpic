"""Quick qualitative check of a PixelGen checkpoint: a few test captions sampled from the raw and the EMA denoiser,
at given CFG scales, with the official shared noise (seed 0 for every prompt) and with per-prompt noise. One PNG grid
per setting. Uses the code on PYTHONPATH (the code the checkpoint was trained with).

python tools/sample_check.py <ckpt> <out_dir> [--n 4] [--cfg 1.0 4.0] [--steps 50]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cfg", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--captions", default="/scratch/gpfs/GROUP/USER/data/gpic/gpic_test_50k.jsonl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.diffusion.flow_matching.sampling import EulerSamplerJiT, ode_step_fn
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    from src.diffusion.base.guidance import simple_guidance_fn
    dev = torch.device("cuda")
    caps = []
    with open(a.captions) as f:
        for line in f:
            d = json.loads(line)
            caps.append(d.get("prompt") or d.get("caption"))
            if len(caps) == a.n:
                break
    te = Qwen3TextEncoder(weight_path="Qwen/Qwen3-1.7B", embed_dim=2048, max_length=300).to(dev)
    with torch.no_grad():
        cond, uncond = te(caps, {})
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    kw = dict(input_size=256, in_channels=3, num_groups=24, hidden_size=1536, num_blocks=16, num_text_blocks=4,
              patch_size=16, txt_embed_dim=2048, txt_max_length=300, bottleneck_dim=256)
    noise_shared = torch.randn((3, 256, 256), generator=torch.Generator().manual_seed(0)).expand(a.n, -1, -1, -1).clone()
    noise_each = torch.stack([torch.randn((3, 256, 256), generator=torch.Generator().manual_seed(1000 + i)) for i in range(a.n)])
    report = {"captions": caps}
    for which in ("ema_denoiser", "denoiser"):
        net = JiT_T2I(**kw)
        net.load_state_dict({k[len(which) + 1:]: v for k, v in sd.items() if k.startswith(which + ".")})
        net = net.to(dev).float().eval()
        for cfg in a.cfg:
            sampler = EulerSamplerJiT(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), timeshift=1.0,
                                      guidance_interval_min=0.0, guidance_interval_max=1.0, step_fn=ode_step_fn,
                                      last_step=None, last_step_fn=ode_step_fn, guidance_fn=simple_guidance_fn,
                                      num_steps=a.steps, guidance=cfg)
            for nname, noise in (("shared", noise_shared), ("each", noise_each)):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    img = sampler(net, noise.to(dev), cond, uncond)
                img = ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                grid = np.concatenate(list(img), axis=1)
                name = f"{which}_cfg{cfg:g}_{nname}.png"
                Image.fromarray(grid).save(os.path.join(a.out, name))
                report[name] = {"pixel_std_across_images": float(img.astype(np.float32).std(axis=0).mean()),
                                "pixel_std_within": float(img.astype(np.float32).std(axis=(1, 2)).mean())}
                print(name, report[name], flush=True)
        del net
        torch.cuda.empty_cache()
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
