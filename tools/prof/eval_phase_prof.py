"""Profile the evaluation phases on one GPU: 50K-image sampling (predict) and gpic-eval DINOv2 feature extraction.

Sampling: the official predict path per batch of 32 captions = text encoder -> 50 Euler steps of the (compiled, fp32
weights, bf16 autocast) EMA denoiser on a CFG batch of 64 -> decode -> uint8 -> PNG. Each stage is timed with CUDA
events; three Euler steps are captured with torch.profiler (shapes, stacks). Random-init weights: same compute.
DINOv2: the gpic_eval backend (PIL bicubic 224 resize in DataLoader workers, ViT-L/14 fp32) over 2048 PNGs, with the
CLI default num_workers=0 and with 32 workers; data wait vs GPU time per batch, plus a profiled window.
usage: python eval_phase_prof.py <out_dir>
"""
import io, json, os, sys, time

import numpy as np
import torch

WT = "/scratch/gpfs/GROUP/USER/project/gpic-wt/perf"
sys.path.insert(0, f"{WT}/baselines/PixelGen"); sys.path.insert(0, f"{WT}/tools"); sys.path.insert(0, f"{WT}/gpic_eval/..")


def ev():
    return torch.cuda.Event(enable_timing=True)


def sampling(out):
    from correctness import build
    from src.models.autoencoder.base import fp2uint8
    from PIL import Image
    te, den, ema, trainer, sampler = build(compile_models=True)
    caps = [json.loads(l)["caption"] for l, _ in zip(open("/scratch/gpfs/GROUP/USER/data/gpic/gpic_test_50k.jsonl"), range(32))]
    res = {}
    ema.eval()
    for it in range(3):  # it 0 compiles; it 2 is steady
        t = {k: (ev(), ev()) for k in ("te", "sample", "decode")}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t["te"][0].record(); cond, uncond = te(caps, {}); t["te"][1].record()
            x = torch.randn(32, 3, 256, 256, device="cuda")
            t["sample"][0].record(); img = sampler(ema, x, cond, uncond); t["sample"][1].record()
            t["decode"][0].record(); u8 = fp2uint8(img); t["decode"][1].record()
        torch.cuda.synchronize()
        arr = u8.permute(0, 2, 3, 1).cpu().numpy() if u8.shape[1] == 3 else u8.cpu().numpy()
        t0 = time.time()
        for a in arr:
            Image.fromarray(a).save(io.BytesIO(), format="PNG")
        png_s = time.time() - t0
        res[f"iter{it}"] = {k: a.elapsed_time(b) for k, (a, b) in t.items()} | {"png_cpu_s_32": png_s}
    # profiled window: 3 Euler steps
    sampler3 = type(sampler)(scheduler=sampler.scheduler, w_scheduler=sampler.w_scheduler, timeshift=1.0,
                             guidance_interval_min=0.0, guidance_interval_max=1.0, step_fn=sampler.step_fn,
                             last_step_fn=sampler.last_step_fn, guidance_fn=sampler.guidance_fn, num_steps=3, guidance=4.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sampler3(ema, x, cond, uncond); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_stack=True, with_flops=True) as p:
            sampler3(ema, x, cond, uncond); torch.cuda.synchronize()
    p.export_chrome_trace(f"{out}/sampling_rank0.json")
    open(f"{out}/sampling_ops.txt", "w").write(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))
    del te, den, ema
    torch.cuda.empty_cache()
    return res


def dino(out):
    from PIL import Image
    from torch.utils.data import DataLoader
    from gpic_eval.backends.dino import build_dinov2, _dino_transform
    from gpic_eval.utils import make_dataset
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "dino_pngs")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(2048):
        Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(f"{d}/{i:05d}.png")
    model = build_dinov2(torch.device("cuda"))
    res = {}
    for nw in (0, 32):
        dl = DataLoader(make_dataset(d, transform=_dino_transform), batch_size=512, num_workers=nw)
        waits, gpus = [], []
        t_last = time.time()
        for i, batch in enumerate(dl):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            waits.append(time.time() - t_last)
            a, b = ev(), ev()
            a.record()
            with torch.no_grad():
                model(x.cuda())
            b.record(); torch.cuda.synchronize()
            gpus.append(a.elapsed_time(b) / 1e3)
            t_last = time.time()
        res[f"workers{nw}"] = {"batches": len(gpus), "data_wait_s": waits, "gpu_s": gpus,
                               "img_per_s": 2048 / (sum(waits) + sum(gpus))}
    x = torch.randn(512, 3, 224, 224, device="cuda")
    with torch.no_grad():
        model(x); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as p:
            model(x); torch.cuda.synchronize()
    p.export_chrome_trace(f"{out}/dino_rank0.json")
    open(f"{out}/dino_ops.txt", "w").write(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
    return res


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    r = {}
    for name, fn in (("sampling", sampling), ("dino", dino)):
        try:
            r[name] = fn(out)
        except Exception as e:
            import traceback; r[name] = {"error": traceback.format_exc()[-2000:]}
        json.dump(r, open(f"{out}/eval_phase.json", "w"), indent=1, default=str)
    print(json.dumps({k: (v if "error" in v else "ok") for k, v in r.items()}, indent=1)[:3000])
