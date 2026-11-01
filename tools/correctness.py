"""Correctness recorder for the speed campaign (parity could not be installed; this is the minimal equivalent).

record: one GPU, fixed real batch, seeds fixed. Builds the official components with the official hyper-parameters
        and records, into one .pt file:
          te_cond / te_uncond   text-encoder output for 32 fixed captions and for ""
          den_out               denoiser output on a fixed (x_t, t, y), before any update
          loss[k], gnorm[k]     the first K training steps (AdamW lr 1e-4 constant, betas .9/.95, clip 1.0, bf16 autocast)
          p_sum/p_abs, ema_*    parameter checksums after K steps
          samples               4 captions, fixed noise, 50 Euler steps, CFG 4.0, uint8 images from the EMA net
derive: python correctness.py derive out/tol.json base1.pt base2.pt base3.pt [kernel_swap.pt ...]
        tolerance per key = 2 x the largest max-abs difference seen between any two of the given runs (floor 1e-6).
compare: python correctness.py compare tol.json base.pt cand.pt  -> PASS/FAIL per key (exit 1 on FAIL)

The code under test is imported from the current worktree (PYTHONPATH), so a candidate that changes the model,
the text encoder, the trainer or the sampler is exercised through the same constructors the config uses.
"""
import argparse, glob, io, json, os, sys, tarfile

import torch

K_STEPS = 10
DATA = "/scratch/gpfs/GROUP/USER/data/gpic/val/gpic_val_00000.tar"
FIXED = "/scratch/gpfs/GROUP/USER/runs/gpic-speed/correctness/fixed_batch.pt"


def fixed_batch(n=32):
    if os.path.exists(FIXED):
        return torch.load(FIXED)
    from PIL import Image
    from src.data.dataset.gpic_dataset import val_transform
    imgs, caps = [], []
    with tarfile.open(DATA) as tar:
        members = {}
        for m in tar:
            if m.isfile():
                members.setdefault(m.name.rsplit(".", 1)[0], {})[m.name.rsplit(".", 1)[1]] = tar.extractfile(m).read()
            if len(members) > n + 4:
                break
    for k in sorted(members)[: n]:
        d = members[k]
        img_key = [e for e in d if e != "json"][0]
        imgs.append(val_transform(Image.open(io.BytesIO(d[img_key])).convert("RGB")))
        caps.append(json.loads(d["json"])["caption"])
    b = {"x": torch.stack(imgs), "captions": caps}
    os.makedirs(os.path.dirname(FIXED), exist_ok=True)
    torch.save(b, FIXED)
    return b


def build(compile_models=True):
    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.diffusion.flow_matching.training_JiT import JiTTrainer
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    from src.diffusion.flow_matching.sampling import EulerSamplerJiT, ode_step_fn
    from src.diffusion.base.guidance import simple_guidance_fn
    torch.manual_seed(0)
    te = Qwen3TextEncoder(weight_path="Qwen/Qwen3-1.7B", embed_dim=2048, max_length=300).cuda()
    den = JiT_T2I(input_size=256, in_channels=3, num_groups=24, hidden_size=1536, num_blocks=16, num_text_blocks=4,
                  patch_size=16, txt_embed_dim=2048, txt_max_length=300, bottleneck_dim=256).cuda()
    trainer = JiTTrainer(scheduler=LinearScheduler(), lognorm_t=True, timeshift=1.0, P_mean=-0.8, P_std=0.8,
                         t_eps=0.05, null_condition_p=0.1)
    sampler = EulerSamplerJiT(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), timeshift=1.0,
                              guidance_interval_min=0.0, guidance_interval_max=1.0, step_fn=ode_step_fn,
                              last_step_fn=ode_step_fn, guidance_fn=simple_guidance_fn, num_steps=50, guidance=4.0)
    import copy
    ema = copy.deepcopy(den).float()
    for p in ema.parameters():
        p.requires_grad_(False)
    if compile_models:
        den.compile()
        ema.compile()
    return te, den, ema, trainer, sampler


def record(out):
    if os.environ.get("CORR_SDPA") == "math":   # kernel-swap envelope: same math, different attention kernel
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    from src.utils import patch_bugs  # noqa: F401  (sets matmul precision like main.py)
    b = fixed_batch()
    te, den, ema, trainer, sampler = build(compile_models=os.environ.get("CORR_COMPILE", "1") == "1")
    rec = {}
    x = b["x"].cuda()
    with torch.no_grad():
        cond, uncond = te(b["captions"], {})
    rec["te_cond"], rec["te_uncond"] = cond.float().cpu(), uncond.float().cpu()

    g = torch.Generator(device="cuda").manual_seed(1)
    t = torch.rand(x.shape[0], device="cuda", generator=g)
    xt = (1 - t.view(-1, 1, 1, 1)) * torch.randn(x.shape, device="cuda", generator=g) + t.view(-1, 1, 1, 1) * x
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rec["den_out"] = den(xt, t, cond).float().cpu()

    # the job's own optimizer settings (CORR_CONFIG, default the bench config of the code under test) and its own
    # EMA callback, so optimizer/EMA changes are exercised, not re-implemented here
    import yaml
    cfg = yaml.safe_load(open(os.environ.get("CORR_CONFIG", "configs_h100/lite_bench.yaml")))
    oa = dict(cfg["model"]["optimizer"]["init_args"])
    opt = torch.optim.AdamW(den.parameters(), lr=oa["lr"], betas=tuple(oa["betas"]), eps=oa["eps"],
                            weight_decay=oa["weight_decay"], fused=oa.get("fused") or None, foreach=oa.get("foreach"))
    from src.callbacks.simple_ema import SimpleEMA
    ema_cb = SimpleEMA(decay=0.9999)
    ema_cb.setup_models(net=den, ema_net=ema)
    losses, gnorms = [], []
    for k in range(K_STEPS):
        torch.manual_seed(100 + k)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = trainer(den, ema, sampler, x, cond, uncond, {})
        res["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
        if hasattr(ema_cb, "_join"):
            ema_cb._join()
        opt.step(); opt.zero_grad(set_to_none=True)
        ema_cb.ema_step()
        torch.cuda.synchronize()
        losses.append(res["loss"].item()); gnorms.append(gn.item())
    rec["loss"], rec["gnorm"] = torch.tensor(losses), torch.tensor(gnorms)
    with torch.no_grad():
        rec["p_sum"] = torch.tensor([p.float().sum().item() for p in den.parameters()])
        rec["p_abs"] = torch.tensor([p.float().abs().sum().item() for p in den.parameters()])
        rec["ema_sum"] = torch.tensor([p.sum().item() for p in ema.parameters()])
        gs = torch.Generator(device="cuda").manual_seed(7)
        noise = torch.randn(4, 3, 256, 256, device="cuda", generator=gs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = sampler(ema, noise, cond[:4], uncond[:4].expand(4, -1, -1))
        rec["samples"] = ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().float()
    torch.save(rec, out)
    print(f"recorded {out}: loss {losses[0]:.5f}->{losses[-1]:.5f}")


def _diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def derive(tol_out, runs):
    recs = [torch.load(r) for r in runs]
    tol = {}
    for key in recs[0]:
        m = max(_diff(recs[i][key], recs[j][key]) for i in range(len(recs)) for j in range(i + 1, len(recs)))
        tol[key] = max(2 * m, 1e-6)
    json.dump({"tol": tol, "runs": runs}, open(tol_out, "w"), indent=1)
    print(json.dumps(tol, indent=1))


def compare(tol_file, base, cand):
    tol = json.load(open(tol_file))["tol"]
    a, b = torch.load(base), torch.load(cand)
    bad = 0
    for key, t in tol.items():
        d = _diff(a[key], b[key])
        ok = d <= t
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {key:10s} maxabs {d:.3e} tol {t:.3e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "derive", "compare"])
    ap.add_argument("args", nargs="+")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.args[0])
    elif a.cmd == "derive":
        derive(a.args[0], a.args[1:])
    else:
        compare(*a.args)
