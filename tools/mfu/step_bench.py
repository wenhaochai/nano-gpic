#!/usr/bin/env python3
"""The ladder's training step without Lightning and without the data path, for kernel work (docs/mfu.md).

The same math as production (configs_h100/lite_ladder.yaml): JiT_T2I at width d, compiled as configure_model does
(nn.Module.compile, Inductor's C++ launcher when TORCHINDUCTOR_CPP_WRAPPER=1), JiTTrainer's x-pred flow-matching loss under
bf16 autocast, global-norm clip 1.0 (foreach), fused AdamW (lr 1e-4, betas 0.9/0.95, wd 0), the EMA as one foreach lerp
on a side stream, joined before the next optimizer step. Batches are synthetic and live on the GPU: 32 images, text
embeddings with a padded length T drawn per rank and step from the measured histogram (docs/ladder_runs.json T_hist,
mean 51.5), the first n tokens of each caption valid (n within the last 8 below T for one sample, the rest spread
below it, as a sorted window gives).
Single GPU: python step_bench.py --d 1280 --nb 14 --nt 3. Under torchrun: DDP as Lightning sets it (bucket_cap_mb 1000,
gradient_as_bucket_view, no buffer broadcast) around a module whose forward is the training step, so the compiled
denoiser runs inside DDP's forward and Dynamo splits its graph at the buckets.
Outputs (--out x.json): per-step start-to-start times (CUDA events), mean / median over the measured steps, MFU, peak
memory; --profile a:b adds a torch.profiler chrome trace of steps [a, b); --record adds per-step loss and grad norm
and a final parameter checksum, for parity checks between variants (same seed, same batches).
"""
import argparse, copy, json, math, os, random, statistics, sys, time

import torch
import torch.distributed as dist
import torch.nn as nn

HIST = {16: 0.084, 24: 0.2741, 32: 0.0958, 40: 0.1019, 48: 0.1609, 56: 0.1358, 64: 0.0384, 72: 0.0046, 80: 0.001,
        88: 0.001, 120: 0.001, 128: 0.0041, 136: 0.0031, 144: 0.0087, 152: 0.0215, 160: 0.0092, 168: 0.022,
        176: 0.0015, 192: 0.001, 200: 0.0036, 208: 0.002, 216: 0.0082, 224: 0.0036, 232: 0.0036, 240: 0.0041,
        248: 0.0015, 256: 0.0005, 264: 0.0005, 272: 0.0005, 280: 0.001, 288: 0.0005, 320: 0.0005}
GF = {640: 104.1, 768: 163.8, 896: 242.1, 1024: 346.9, 1280: 622.4, 1536: 1024.9, 1792: 1554.8}
SIZES = {640: (9, 2), 768: (10, 2), 896: (11, 2), 1024: (12, 3), 1280: (14, 3), 1536: (16, 4), 1792: (18, 4)}
PEAK = 989.4e12


def make_lengths(T, B, rng):
    """Valid caption lengths of one batch padded to T: one in (T-8, T], the others below it."""
    top = rng.randint(max(1, T - 7), T)
    lens = [top] + [rng.randint(max(1, min(top, T - 16)), top) for _ in range(B - 1)]
    rng.shuffle(lens)
    return lens


class Step(nn.Module):
    """The training step as one module (Lightning's DDP wraps the LightningModule and calls training_step)."""

    def __init__(self, net, trainer):
        super().__init__()
        self.net, self.trainer = net, trainer

    def forward(self, x, y, mask):
        return self.trainer(self.net, None, None, x, y, None, {"y_mask": mask})["loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--nb", type=int)
    ap.add_argument("--nt", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60, help="measured steps")
    ap.add_argument("--warm", type=int, default=15, help="steps before the measured ones, after the compile warm-up")
    ap.add_argument("--T", default="hist", help="'hist' (per rank and step from the histogram), or a fixed padded length")
    ap.add_argument("--same-T", action="store_true", help="all ranks draw the same T per step (no rank imbalance)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="no autocast (for bit-exact checks)")
    ap.add_argument("--profile", default=None, help="a:b, profile measured steps [a, b)")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nb, nt = (a.nb, a.nt) if a.nb else SIZES[a.d]

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if ddp:
        dist.init_process_group("nccl", device_id=dev)
    world = dist.get_world_size() if ddp else 1

    sys.path.insert(0, os.environ.get("PIXGEN_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "PixelGen")))
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.diffusion.flow_matching.training_JiT import JiTTrainer, constant
    from src.diffusion.flow_matching.scheduling import LinearScheduler

    torch.manual_seed(a.seed)
    net = JiT_T2I(input_size=256, in_channels=3, num_groups=a.d // 64, hidden_size=a.d, num_blocks=nb, num_text_blocks=nt,
                  patch_size=16, txt_embed_dim=2048, txt_max_length=320, bottleneck_dim=256).to(dev)
    ema = copy.deepcopy(net).float()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if not a.no_compile:
        net.compile()
    trainer = JiTTrainer(scheduler=LinearScheduler(), loss_weight_fn=constant, lognorm_t=True, timeshift=1.0,
                         P_mean=-0.8, P_std=0.8, t_eps=0.05, null_condition_p=0.0)
    step_mod = Step(net, trainer)
    model = step_mod
    if ddp:
        model = nn.parallel.DistributedDataParallel(step_mod, device_ids=[local], bucket_cap_mb=1000,
                                                    gradient_as_bucket_view=True, broadcast_buffers=False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
    ema_params, net_params = list(ema.parameters()), list(net.parameters())
    ema_stream = torch.cuda.Stream(device=dev)

    # batches: a pool on the GPU, cycled
    rng = random.Random(a.seed * 1000 + (0 if a.same_T else rank))
    Ts_all, Ps = list(HIST), [HIST[t] for t in HIST]
    total = a.warm + a.steps
    pool_n = 8
    g = torch.Generator(device=dev).manual_seed(a.seed * 7919 + rank)
    xs = [torch.rand(a.batch, 3, 256, 256, device=dev, generator=g) * 2 - 1 for _ in range(pool_n)]
    emb = torch.randn(a.batch, 320, 2048, device=dev, generator=g).to(torch.bfloat16)

    def batch(i, T):
        """Built before the timed loop: a pageable host-to-device copy inside it would stall the host on the GPU."""
        lens = make_lengths(T, a.batch, rng)
        mask = torch.arange(T, device=dev)[None] < torch.tensor(lens, device=dev)[:, None]
        y = emb[:, :T] * mask[..., None]
        return xs[i % pool_n], y, mask

    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if not a.fp32 else (lambda: torch.autocast("cuda", enabled=False))

    def train_step(x, y, mask, rec=None):
        with ctx():
            loss = model(x, y, mask)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        torch.cuda.current_stream().wait_stream(ema_stream)   # SimpleEMA._join before the optimizer step
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ema_stream), torch.no_grad():
            torch._foreach_lerp_(ema_params, net_params, 1.0 - 0.9999)
        if rec is not None:
            rec.append((loss.detach(), gn.detach()))

    # compile warm-up as CompileWarmup: T = 64, 128 (dynamic T), 320, 192
    t0 = time.time()
    for T in (64, 128, 320, 192):
        x, y, mask = batch(0, T)
        train_step(x, y, mask)
    torch.cuda.synchronize()
    warm_s = time.time() - t0
    if a.T == "hist":
        Tseq = [rng.choices(Ts_all, Ps)[0] for _ in range(total)]
    else:
        Tseq = [int(a.T)] * total
    batches = [batch(i, Tseq[i]) for i in range(total)]
    torch.cuda.synchronize()
    torch.manual_seed(a.seed + 1)
    rec = [] if a.record else None
    evs = []
    prof = None
    pa = tuple(int(v) for v in a.profile.split(":")) if a.profile else None
    torch.cuda.reset_peak_memory_stats()
    for i in range(total):
        j = i - a.warm
        if pa and j == pa[0]:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        x, y, mask = batches[i]
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        evs.append(e)
        if prof is not None:
            with torch.profiler.record_function(f"ProfilerStep#{j}"):
                train_step(x, y, mask, rec if j >= 0 else None)
        else:
            train_step(x, y, mask, rec if j >= 0 else None)
        if pa and j == pa[1] - 1:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            if a.out:
                prof.export_chrome_trace(a.out.replace(".json", f"_trace_rank{rank}.json"))
            prof = None
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    evs.append(end)
    torch.cuda.synchronize()
    ms = [evs[i].elapsed_time(evs[i + 1]) for i in range(len(evs) - 1)][a.warm:]
    keep = [m for k, m in enumerate(ms) if not (pa and pa[0] - 1 <= k <= pa[1])]
    mean = statistics.mean(keep)
    res = {"d": a.d, "nb": nb, "nt": nt, "params": n_params, "world": world, "rank": rank, "T": a.T, "same_T": a.same_T,
           "steps": len(keep), "mean_ms": mean, "median_ms": statistics.median(keep), "min_ms": min(keep),
           "max_ms": max(keep), "compile_warm_s": warm_s, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
           "mean_T": statistics.mean(Tseq[a.warm:]), "step_ms": ms,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("GPIC_", "TORCHINDUCTOR", "NCCL_"))},
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev)}
    if a.d in GF:
        res["mfu"] = GF[a.d] * 1e9 * a.batch / (mean / 1e3) / PEAK
    if rec is not None:
        res["loss"] = [float(l) for l, _ in rec]
        res["grad_norm"] = [float(n) for _, n in rec]
        torch.cuda.synchronize()
        res["param_sum"] = float(sum(p.double().sum() for p in net_params))
        res["param_abs"] = float(sum(p.double().abs().sum() for p in net_params))
        res["ema_sum"] = float(sum(p.double().sum() for p in ema_params))
    if rank == 0:
        print(f"d{a.d} {nb}+{nt} world {world} T={a.T}{' same' if a.same_T else ''}: mean {mean:.2f} ms, median "
              f"{res['median_ms']:.2f}, MFU {100 * res.get('mfu', float('nan')):.1f}%, peak {res['peak_alloc_gb']:.1f} GiB, "
              f"warm-up {warm_s:.0f} s", flush=True)
    if a.out and rank == 0:
        json.dump(res, open(a.out, "w"))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
