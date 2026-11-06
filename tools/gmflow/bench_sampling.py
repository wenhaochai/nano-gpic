#!/usr/bin/env python3
"""Sampling throughput on 8 GPUs (docs/gmflow_hn.md, N2): images per second of the samplers alone, no FD, for the d640 C0/8 checkpoints.

Settings: the point model with the official Euler at 50, 16, 8 and 4 calls; the K = 8 model with GM-ODE at 8 and 4 calls (ceil(128 / calls) sub-steps), not fused and
fused (gm_ode_triton.py). As in the evaluation (jobs/eval_gmflow.sh): EMA weights in fp32, the denoiser compiled (as configure_model), bf16 autocast around the
sampler, masked captions (lite_ladder_sample.yaml), batch 32 per GPU, no CFG.
Prompts: the first P x world prompts of gpic_test_50k.jsonl, P per rank, each with its own fixed noise; the captions are encoded once before any timing (the text
encoder is the same for every sampler and is not timed). Per setting: one warm-up pass over the rank's prompts (compiles every caption length), then timed passes until
the slowest rank has spent at least --min-seconds; images / s = passes x P x world / the slowest rank's time.
Checks (review_gmflow_hn F4), all on an uncompiled fp32 copy of the K = 8 net with autocast and TF32 off, on each rank's first 32 prompts:
(1) per call, the fused kernel against the PyTorch sub-steps (gm_ode_substeps_ref) on a real K = 8 output, at the first call (sigma 1 -> 1 - 1/calls) and at the last
call (t = 1 - 1/calls, sigma 1/calls -> 0, where the clamps act), max |diff| < 1e-4; (2) end to end, GM-ODE fused against not fused at 8 and 4 calls, max |diff| < 1e-4.
The bf16 end-to-end difference of the timed runs (the bf16 network amplifies fp32 rounding along the path) is in the JSON as a report only; it fails nothing.
Also: the sub-steps of one call alone (fused and not), in ms, on a real output.
usage (torchrun, one process per GPU): bench_sampling.py --point <ckpt> --gm <ckpt> --out <json> [--prompts 1024] [--min-seconds 60]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist

from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.sampling import EulerSamplerJiT
from src.diffusion.flow_matching.sampling_gmflow import GMODESampler
from src.diffusion.flow_matching.scheduling import LinearScheduler

TEST_JSONL = "/scratch/gpfs/GROUP/USER/data/gpic/gpic_test_50k.jsonl"


def build_net(ckpt, gm_k, d=640, nb=9, nt=2):
    from src.models.transformer.JiT_T2I import JiT_T2I
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    sd = {k[len("ema_denoiser."):]: v for k, v in sd.items() if k.startswith("ema_denoiser.")}
    out = sd["final_layer.linear.weight"].shape[0]
    assert out == (gm_k * 4 * 256 if gm_k else 768), f"{ckpt}: final layer {out} outputs, K = {gm_k}"
    net = JiT_T2I(input_size=256, in_channels=3, num_groups=d // 64, hidden_size=d, num_blocks=nb, num_text_blocks=nt, patch_size=16, txt_embed_dim=2048,
                  txt_max_length=320, bottleneck_dim=256, gm_k=gm_k)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected and all("pos_embed" in m for m in missing), (missing, unexpected)
    return net.cuda().float().eval().requires_grad_(False)


def make_sampler(kind, calls, fused=False):
    kw = dict(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(), guidance_fn=simple_guidance_fn, num_steps=calls, guidance=1.0, timeshift=1.0)
    if kind == "euler":
        return EulerSamplerJiT(**kw)
    return GMODESampler(substeps=max(2, -(-128 // calls)), fused=fused, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True); ap.add_argument("--gm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=1024); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--min-seconds", type=float, default=60)
    a = ap.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))   # before the process group (review_gmflow_hn F11)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda")
    lines = open(TEST_JSONL).readlines()[rank * a.prompts:(rank + 1) * a.prompts]
    caps = [json.loads(l)["caption"] for l in lines]
    assert len(caps) == a.prompts, len(caps)

    from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
    te = Qwen3TextEncoder(weight_path="Qwen/Qwen3-1.7B", embed_dim=2048, max_length=320, text_mode="masked", pad_multiple=8, attn_implementation="sdpa").to(dev).eval()
    batches = []
    for b0 in range(0, a.prompts, a.batch):
        meta = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y, _ = te(caps[b0:b0 + a.batch], meta)
        g = torch.Generator(device=dev).manual_seed(rank * 1_000_003 + b0)
        noise = torch.randn(len(caps[b0:b0 + a.batch]), 3, 256, 256, generator=g, device=dev)
        batches.append((noise, y, meta["y_mask"]))
    del te
    torch.cuda.empty_cache()

    def sync():
        torch.cuda.synchronize(); dist.barrier()

    def run_pass(sampler, f):
        outs = []
        for noise, y, mask in batches:
            g = lambda x, t, c, _m=mask, **kw: f(x, t, c, y_mask=_m, **kw)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(sampler(g, noise, y, None))
        return outs

    results = dict(prompts_per_rank=a.prompts, world=world, batch=a.batch, min_seconds=a.min_seconds, gpu=torch.cuda.get_device_name(), settings=[])
    nets = {"point": (a.point, 0), "gm8": (a.gm, 8)}
    first_out = {}
    for name, settings in (("point", [("euler", 50, False), ("euler", 16, False), ("euler", 8, False), ("euler", 4, False)]),
                           ("gm8", [("gm_ode", 8, False), ("gm_ode", 8, True), ("gm_ode", 4, False), ("gm_ode", 4, True)])):
        net = build_net(*nets[name])
        net.compile()   # as LightningModel.configure_model compiles ema_denoiser
        for kind, calls, fused in settings:
            s = make_sampler(kind, calls, fused)
            sync(); t0 = time.perf_counter()
            outs = run_pass(s, net)   # warm-up: every caption length compiles here
            sync(); warm = time.perf_counter() - t0
            if kind == "gm_ode":
                first_out[(calls, fused)] = outs[0].float().cpu()
            passes, t0 = 0, time.perf_counter()
            while True:
                run_pass(s, net)
                passes += 1
                torch.cuda.synchronize()
                el = torch.tensor([time.perf_counter() - t0], device=dev)
                dist.all_reduce(el, op=dist.ReduceOp.MAX)
                if el.item() >= a.min_seconds:
                    break
            imgs = passes * a.prompts * world
            row = dict(model=name, ckpt=nets[name][0], sampler=kind, calls=calls, substeps=getattr(s, "substeps", None), fused=fused, passes=passes, images=imgs,
                       seconds=el.item(), images_per_s=imgs / el.item(), images_per_s_per_gpu=imgs / el.item() / world, warmup_s=warm)
            results["settings"].append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
        if name == "gm8":   # the sub-steps alone (timing) and the first-call check, on a real output at the first call
            from src.diffusion.flow_matching.gm_ode_triton import gm_ode_substeps_fused, gm_ode_substeps_ref
            from src.diffusion.flow_matching.gmflow import logstd_x
            noise, y, mask = batches[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gm = net(noise, torch.zeros(noise.shape[0], device=dev), y, y_mask=mask, gm_mode="full")
            gm = {k: v.float() for k, v in gm.items()}
            for calls in (8, 4):
                n = max(2, -(-128 // calls))
                sig = torch.tensor([1 - (1 / calls) * j / n for j in range(n + 1)], device=dev)
                s_x = logstd_x(gm, torch.ones(noise.shape[0], 1, 1, 1, 1, device=dev)).exp()
                ref = gm_ode_substeps_ref(gm["means"], gm["logweights"], s_x, noise, sig)
                fu = gm_ode_substeps_fused(gm["means"], gm["logweights"], s_x, noise, sig)
                d = (ref - fu).abs()
                ms = {}
                for label, fn in (("pytorch", gm_ode_substeps_ref), ("fused", gm_ode_substeps_fused)):
                    for _ in range(3):
                        fn(gm["means"], gm["logweights"], s_x, noise, sig)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for _ in range(20):
                        fn(gm["means"], gm["logweights"], s_x, noise, sig)
                    torch.cuda.synchronize(); ms[label] = (time.perf_counter() - t0) / 20 * 1000
                chk = dict(call="first, bf16 net output", calls=calls, substeps=n, max_abs_diff=d.max().item(), mean_abs_diff=d.mean().item(),
                           ms_per_call_pytorch=ms["pytorch"], ms_per_call_fused=ms["fused"], batch=noise.shape[0])
                results.setdefault("per_call", []).append(chk)
                if rank == 0:
                    print(json.dumps(chk), flush=True)
        if name == "gm8":   # F4: fp32 checks on an uncompiled copy, autocast and TF32 off
            from src.diffusion.flow_matching.gm_ode_triton import gm_ode_substeps_fused, gm_ode_substeps_ref
            from src.diffusion.flow_matching.gmflow import logstd_x
            tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            net32 = build_net(*nets[name])
            noise, y, mask = batches[0]
            y32 = y.float()
            g32 = lambda x, t, c, _m=mask, **kw: net32(x, t, c, y_mask=_m, **kw)
            results["fp32_end_to_end"], B = [], noise.shape[0]
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                for calls in (8, 4):
                    su, sf = make_sampler("gm_ode", calls, False), make_sampler("gm_ode", calls, True)
                    xs = su._impl_sampling(g32, noise, y32, None)[0]
                    xf = sf._impl_sampling(g32, noise, y32, None)[0][-1]
                    d = (xs[-1] - xf).abs()
                    results["fp32_end_to_end"].append(dict(calls=calls, substeps=su.substeps, prompts=B, max_abs_diff=d.max().item(), mean_abs_diff=d.mean().item()))
                    t_last = su.timesteps[-2].item()   # = 1 - 1 / calls
                    gm = {k: v.float() for k, v in net32(xs[-2], torch.full((B,), t_last, device=dev), y32, y_mask=mask, gm_mode="full").items()}
                    s0 = torch.tensor(1 - t_last, device=dev)
                    sig = su._substep_sigmas(s0.view(1, 1, 1, 1), torch.zeros(1, 1, 1, 1, device=dev))
                    s_x = logstd_x(gm, s0.expand(B).view(B, 1, 1, 1, 1)).exp()
                    d = (gm_ode_substeps_ref(gm["means"], gm["logweights"], s_x, xs[-2], sig) - gm_ode_substeps_fused(gm["means"], gm["logweights"], s_x, xs[-2], sig)).abs()
                    results["per_call"].append(dict(call=f"last (t = {t_last:.4f}), fp32 net", calls=calls, substeps=su.substeps, max_abs_diff=d.max().item(),
                                                    mean_abs_diff=d.mean().item(), batch=B))
                    if rank == 0:
                        print(json.dumps(results["fp32_end_to_end"][-1]), json.dumps(results["per_call"][-1]), flush=True)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
            del net32
        del net
        torch.cuda.empty_cache()

    e2e = []
    for calls in (8, 4):
        a_, b_ = first_out[(calls, False)], first_out[(calls, True)]
        d = (a_ - b_).abs()
        u8 = lambda z: ((z.clamp(-1, 1) + 1) * 127.5).round()
        e2e.append(dict(calls=calls, max_abs_diff=d.max().item(), mean_abs_diff=d.mean().item(), uint8_share_diff_gt1=((u8(a_) - u8(b_)).abs() > 1).float().mean().item()))
    gathered = [None] * world
    dist.all_gather_object(gathered, e2e)
    results["end_to_end_bf16_report_only"] = gathered
    mine = dict(per_call=results["per_call"], fp32_end_to_end=results["fp32_end_to_end"])
    allr = [None] * world
    dist.all_gather_object(allr, mine)
    results["per_call"], results["fp32_end_to_end"] = [c for r in allr for c in r["per_call"]], [c for r in allr for c in r["fp32_end_to_end"]]
    pc_ok = all(c["max_abs_diff"] < 1e-4 for c in results["per_call"])
    e2e_ok = all(c["max_abs_diff"] < 1e-4 for c in results["fp32_end_to_end"])
    results["checks"] = {"per_call_first_and_last_max_lt_1e-4": pc_ok, "fp32_end_to_end_max_lt_1e-4": e2e_ok}   # string keys: "1e-4" is not a Python name
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
        print(json.dumps(results["checks"]), json.dumps(gathered[0]), flush=True)
        print(f"wrote {a.out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if (pc_ok and e2e_ok) else 1)


if __name__ == "__main__":
    main()
