#!/usr/bin/env python3
"""Test of src/callbacks/launch_fingerprint.py in a real Lightning Trainer on the CPU (no GPU), before
jobs/smoke_launch.sh reads it on 8 GPUs:
  a  tensor_sums and walk: equal tensors give equal sums; a changed, a moved and two swapped elements, another shape and
     another dtype change them; the dtypes of a run (bf16, float32, float64, integers, bool, complex, no element, no
     dimension); a DTensor (FSDP2's parameters) gives its local shard's sums; a changed caption or two swapped samples
     change a batch's hash
  b  one fit with the callback: one start line and two batch lines that tools/mfu/check_launch.py's reader parses, with
     every field the checker reads; the times are in order and the process start agrees with ps; the EMA slot's copy in
     configure_model is timed, called once, and the module's own function is back in place
  c  a second fit from the same seed prints the same hashes; another seed changes the weights and the random streams
     and nothing else
  d  no side effect: a fit without the callback ends with the same weights, optimizer state and random streams
  e  each hash sees its own part of the state: one changed weight, buffer, EMA weight, optimizer moment, learning rate,
     scheduler step or random draw changes exactly its field
  f  a resumed fit: the start line has the checkpoint's step, the batch lines count from 0 at that step, and the
     optimizer state is no longer a fresh run's
usage: python test_launch_fingerprint.py <scratch dir>"""
import contextlib
import copy
import io
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../../../baselines/PixelGen"))
T0 = time.time()
os.environ["GPIC_LAUNCH_T0"] = repr(T0)   # the launcher's start: read when the callback's module is imported
OMP_AT_START = os.environ.get("OMP_NUM_THREADS", "unset")
os.environ.pop("GPIC_VEC_ISA_MEMO", None)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import lightning.pytorch as pl  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import check_launch  # noqa: E402
import src.utils.patch_bugs  # noqa: E402,F401  (as main.py: the memo's counts live there)
from src.callbacks import launch_fingerprint as fp  # noqa: E402

BAD = []
COPIES = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        BAD.append(what)


def copy_params(src_model, dst_model):   # the name configure_model calls, as src.lightning_model_gpic's
    for s, d in zip(src_model.parameters(), dst_model.parameters()):
        d.data.copy_(s.data)
    COPIES.append(1)


class Data(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"x": torch.randn(8, generator=g), "cap": f"caption {i}", "meta": {"id": i, "size": (i, 2 * i)}}


class Toy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.denoiser = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Linear(16, 8))
        self.ema_denoiser = copy.deepcopy(self.denoiser)

    def configure_model(self):
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)
        for p in self.ema_denoiser.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        noise = torch.randn_like(x) * (0.1 + 0.01 * np.random.rand() + 0.01 * random.random())   # the three CPU streams
        return (self.denoiser(x + noise) - x).pow(2).mean()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.denoiser.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 4))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def fit(scratch, seed, steps, with_fp=True, ckpt=None):
    """One fit; the callback's lines, the model and the trainer."""
    pl.seed_everything(seed, verbose=False)
    model = Toy()
    cb = [fp.LaunchFingerprint()] if with_fp else []
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=steps, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, callbacks=cb, default_root_dir=scratch)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trainer.fit(model, DataLoader(Data(), batch_size=4, shuffle=True, num_workers=0), ckpt_path=ckpt)
    return buf.getvalue(), model, trainer, (cb[0] if cb else None)


def read(scratch, name, text):
    log = os.path.join(scratch, name + ".log")
    with open(log, "w") as f:
        f.write("Epoch 0:   0%|   | 0/4 [00:00<?, ?it/s]\r" + text)   # a progress bar's text before the first marker
    a = {"name": name, "log": log, "n": "1", "run": scratch, "bench": os.path.join(scratch, "no_bench")}
    check_launch.read_arm(a)
    return a


def end_state(model, trainer):
    return fp.state_fields(model.denoiser, model.ema_denoiser, trainer.optimizers,
                           [c.scheduler for c in trainer.lr_scheduler_configs])


def part_a(scratch):
    ts = fp.tensor_sums
    g = torch.Generator().manual_seed(0)
    t = torch.randn(5, 7, generator=g)
    check(ts(t) == ts(t.clone()), "a: equal tensors give equal sums")
    u = t.clone(); u[2, 3] = torch.nextafter(u[2, 3], torch.tensor(9.0))
    check(ts(u) != ts(t), "a: one element changed by the smallest step")
    u = t.clone(); u[1, 2], u[4, 0] = t[4, 0], t[1, 2]
    check(ts(u) != ts(t) and ts(u)[2] == ts(t)[2], "a: two swapped elements (the plain sum is equal, the weighted one is not)")
    check(ts(t.roll(1, 1)) != ts(t), "a: every element moved by one place")
    check(ts(t.reshape(7, 5)) != ts(t), "a: another shape")
    check(ts(t.double()) != ts(t), "a: another dtype")
    check(ts(t.t().contiguous().t()) == ts(t) and ts(t.t()) == ts(t.t().contiguous()), "a: a view gives its own elements' sums")
    for dt in (torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64):
        x = (torch.randn(3, 4, generator=g) * 50).to(dt)
        y = x.clone(); y[1, 1] = y[1, 1] + 1
        check(ts(x) == ts(x.clone()) and ts(x) != ts(y), f"a: {str(dt).split('.')[1]}: equal, and one changed element")
    x = torch.tensor([True, False, True]); y = torch.tensor([True, True, False])
    check(ts(x) == ts(x.clone()) and ts(x) != ts(y), "a: bool")
    x = torch.complex(torch.randn(4, generator=g), torch.randn(4, generator=g)); y = x.clone(); y[2] = y[2].conj()
    check(ts(x) == ts(x.clone()) and ts(x) != ts(y), "a: complex")
    check(ts(torch.zeros(0, 3)) == ((0, 3), "torch.float32", 0, 0) and ts(torch.tensor(1.5))[0] == (), "a: no element, no dimension")
    check(ts(torch.tensor([0.0])) != ts(torch.tensor([-0.0])), "a: 0.0 and -0.0 are different bits")
    big = torch.arange((1 << 24) + 5, dtype=torch.int32)   # more than one block of the loop
    b2 = big.clone(); b2[-1], b2[3] = big[3], big[-1]
    check(ts(big) == ts(big.clone()) and ts(big) != ts(b2), "a: a swap across the loop's blocks")
    p = nn.Parameter(t.clone())
    check(ts(p) == ts(t) and p.requires_grad, "a: a parameter gives its tensor's sums")

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    try:
        from torch.distributed.tensor import DTensor, Shard, distribute_tensor
    except ImportError:   # before torch 2.5
        from torch.distributed._tensor import DTensor, Shard, distribute_tensor
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    d = distribute_tensor(t.clone(), mesh, [Shard(0)])
    dp = nn.Parameter(d)
    check(isinstance(d, DTensor) and isinstance(dp, DTensor) and ts(d) == ts(t) and ts(dp) == ts(t),
          "a: a DTensor and a DTensor parameter give the local shard's sums")
    m = nn.Linear(7, 5); m.weight = dp
    f1 = fp.digest([(n, ts(q)) for n, q in m.named_parameters()])
    with torch.no_grad():
        dp.to_local()[0, 0] += 1.0
    check(fp.digest([(n, ts(q)) for n, q in m.named_parameters()]) != f1, "a: a changed element of a DTensor parameter's shard")
    dist.destroy_process_group()

    batch = {"x": t.clone(), "cap": ["a dog", "a cat"], "meta": {"id": torch.tensor([3, 4]), "size": [torch.tensor([1, 2]), None]},
             "n": 2, "w": 0.5, "arr": np.arange(3)}
    h = fp.digest(fp.walk(batch, []))
    b = copy.deepcopy(batch); b["cap"][1] = "a cow"
    check(fp.digest(fp.walk(copy.deepcopy(batch), [])) == h and fp.digest(fp.walk(b, [])) != h, "a: a batch: equal, and one changed caption")
    b = copy.deepcopy(batch); b["cap"].reverse()
    check(fp.digest(fp.walk(b, [])) != h, "a: a batch: two swapped captions")
    b = copy.deepcopy(batch); b["meta"]["id"] = torch.tensor([4, 3])
    check(fp.digest(fp.walk(b, [])) != h, "a: a batch: two swapped ids")
    b = copy.deepcopy(batch); b["meta"]["size"][1] = 0
    check(fp.digest(fp.walk(b, [])) != h, "a: a batch: None against 0")
    b = copy.deepcopy(batch); b["arr"][0] = 7
    check(fp.digest(fp.walk(b, [])) != h, "a: a batch: a changed numpy element")
    opt_state = {t: {"exp_avg": t.clone(), "step": torch.tensor(3.0)}}   # an optimizer's state is keyed by its parameters
    o1 = fp.digest(fp.walk(opt_state, []))
    opt_state2 = {t.clone(): {"exp_avg": t.clone() + 1e-3, "step": torch.tensor(3.0)}}
    check(fp.digest(fp.walk({t.clone(): {"exp_avg": t.clone(), "step": torch.tensor(3.0)}}, [])) == o1
          and fp.digest(fp.walk(opt_state2, [])) != o1, "a: an optimizer state: equal for another tensor key, changed by its value")


def part_bcdf(scratch):
    etimes = int(subprocess.check_output(["ps", "-o", "etimes=", "-p", str(os.getpid())]).split()[0])
    start_ps = time.time() - etimes - T0   # this process's start in the callback's clock, to 1 s
    orig = sys.modules["__main__"].copy_params
    COPIES.clear()
    text, model, trainer, cb = fit(scratch, 0, 3)
    end_with = end_state(model, trainer)   # now: the random streams are the process's
    a = read(scratch, "fit_a", text)
    lines = [l for l in text.splitlines() if "[launch-fp]" in l]
    check(len(lines) == 3 and all(l.startswith("[launch-fp] rank 0 ") for l in lines) and text.endswith("\n"),
          f"b: one start line and two batch lines, each a whole line ({len(lines)} lines)")
    check(sorted(a["start"]) == [0] and sorted(a["batch"]) == [(0, 0), (0, 1)] and a["dup"] == 0, "b: the checker's reader finds them once")
    s = a["start"].get(0, {})
    need = [f for f in check_launch.HARD + check_launch.SOFT if f not in ("rng_cuda", "loss0", "loss1")]
    missing = [f for f in need if check_launch.field(a, f, 0) is None]
    check(not missing and "rng_cuda" not in s, f"b: every field the checker compares is there, no GPU stream on the CPU (missing: {missing})")
    check(check_launch.memo(a, 0) == (0, 0, 0, 0), f"b: the memo's counts parse ({s.get('memo')})")
    check(s.get("step") == "0" and [a["batch"][(0, i)][0] for i in (0, 1)] == [0, 1], "b: step 0, then the batches of steps 0 and 1")
    check(s.get("thr0") == str(fp._THR0) and s.get("omp0") == fp._OMP0 == OMP_AT_START
          and s.get("threads") == f"{torch.get_num_threads()}/{torch.get_num_interop_threads()}"
          and s.get("omp") == os.environ.get("OMP_NUM_THREADS", "unset") and s.get("cpus") == str(len(os.sched_getaffinity(0))),
          f"b: thread counts before fit and at the first step (thr0={s.get('thr0')} omp0={s.get('omp0')} threads={s.get('threads')} "
          f"omp={s.get('omp')} cpus={s.get('cpus')})")
    ts = [check_launch.t(a, 0, k) for k in ("proc", "init", "setup", "fit", "train")] + [a["batch"][(0, i)][2] for i in (0, 1)]
    check(all(x == x for x in ts) and ts == sorted(ts) and ts[0] < 0 < ts[1], f"b: the times are in order, the process start before T0 ({ts})")
    check(abs(ts[0] - start_ps) < 1.6, f"b: the process start agrees with ps ({ts[0]:.2f} against {start_ps:.2f} s)")
    try:
        cs, cf = float(s.get("copy_s")), int(s.get("copy_flt"))
    except (TypeError, ValueError):
        cs = cf = None
    check(cs is not None and 0 <= cs < 5 and cf >= 0 and len(COPIES) == 1, f"b: the EMA slot's copy is timed and called once (copy_s={s.get('copy_s')}, {len(COPIES)} calls)")
    check(sys.modules["__main__"].copy_params is orig and not hasattr(sys.modules["__main__"].copy_params, "_fp_inner"),
          "b: the module's own copy_params is back in place")
    check(s.get("isa", "").startswith(("Vec", "none")), f"b: the vector sets ({s.get('isa')})")
    try:
        check(0 <= float(s.get("fp_s")) < 60, f"b: the fingerprint's own time ({s.get('fp_s')} s)")
    except (TypeError, ValueError):
        check(False, f"b: the fingerprint's own time ({s.get('fp_s')})")

    hashed = ("step", "den", "den_buf", "ema", "ema_buf", "opt", "sched", "rng_cpu", "rng_np", "rng_py", "isa", "batch0", "batch1")
    text2, model2, trainer2, _ = fit(scratch, 0, 3)
    a2 = read(scratch, "fit_a2", text2)
    diff = [f for f in hashed if check_launch.field(a, f, 0) != check_launch.field(a2, f, 0)]
    check(not diff, f"c: a second fit from the same seed prints the same hashes (other: {diff})")
    text3, _, _, _ = fit(scratch, 1, 3)
    a3 = read(scratch, "fit_seed1", text3)
    diff = sorted(f for f in hashed if check_launch.field(a, f, 0) != check_launch.field(a3, f, 0))
    check(diff == sorted(("den", "ema", "rng_cpu", "rng_np", "rng_py", "batch0", "batch1")),
          f"c: another seed changes the weights, the EMA slot, the streams and the batches, nothing else (changed: {diff})")
    check(s.get("den") == s.get("ema") and s.get("den_buf") == s.get("ema_buf") and end_with["den"] != end_with["ema"]
          and end_with["ema"] == s.get("ema"),
          "c: the EMA slot starts with the denoiser's weights and keeps them (this toy has no EMA update)")

    _, model4, trainer4, _ = fit(scratch, 0, 3, with_fp=False)
    end_without = end_state(model4, trainer4)
    diff = [k for k in end_with if end_with[k] != end_without[k]]
    check(not diff and trainer.global_step == trainer4.global_step == 3, f"d: the same end state with and without the callback (other: {diff})")
    check(end_with["den"] != s.get("den") and end_with["opt"] != s.get("opt") and end_with["sched"] != s.get("sched"),
          "d: three steps changed the weights, the optimizer and the scheduler")

    # e: each hash sees its own part of the state
    opt, sched = trainer.optimizers[0], trainer.lr_scheduler_configs[0].scheduler
    base = end_state(model, trainer)

    def changed(what, fn, want):
        fn()
        now = end_state(model, trainer)
        got = sorted(k for k in now if now[k] != base[k])
        check(got == sorted(want), f"e: {what} changes {sorted(want)} (changed: {got})")
        base.clear(); base.update(now)

    def bump(tensor):
        with torch.no_grad():
            flat = tensor.view(-1)
            flat[flat.numel() // 2] = torch.nextafter(flat[flat.numel() // 2], torch.tensor(1e9))

    changed("one weight", lambda: bump(model.denoiser[2].weight), ["den"])
    changed("one EMA weight", lambda: bump(model.ema_denoiser[0].bias), ["ema"])
    changed("one buffer", lambda: bump(model.denoiser[1].running_mean), ["den_buf"])
    changed("one EMA buffer", lambda: bump(model.ema_denoiser[1].running_var), ["ema_buf"])
    changed("one optimizer moment", lambda: bump(opt.state[model.denoiser[0].weight]["exp_avg"]), ["opt"])
    changed("a learning rate", lambda: opt.param_groups[0].__setitem__("lr", 0.123), ["opt"])
    changed("a scheduler step", lambda: sched.__setattr__("last_epoch", sched.last_epoch + 1), ["sched"])
    changed("one torch draw", lambda: torch.rand(1), ["rng_cpu"])
    changed("one numpy draw", lambda: np.random.rand(), ["rng_np"])
    changed("one Python draw", lambda: random.random(), ["rng_py"])

    # f: a resumed fit
    _, model5, trainer5, _ = fit(scratch, 0, 3, with_fp=False)
    ckpt = os.path.join(scratch, "toy_step3.ckpt")
    trainer5.save_checkpoint(ckpt)
    COPIES.clear()
    text6, model6, trainer6, _ = fit(scratch, 0, 5, ckpt=ckpt)
    a6 = read(scratch, "fit_resume", text6)
    s6 = a6["start"].get(0, {})
    check(s6.get("step") == "3" and sorted(a6["batch"]) == [(0, 0), (0, 1)] and [a6["batch"][k][0] for k in sorted(a6["batch"])] == [3, 4],
          f"f: a resumed fit starts at the checkpoint's step and its two batch lines are steps 3 and 4 (step={s6.get('step')}, {sorted(a6['batch'].items())})")
    diff = [k for k in ("den", "den_buf", "ema", "ema_buf", "opt", "sched") if s6.get(k) != end_with[k]]
    check(not diff, f"f: its weights, buffers, EMA slot, optimizer and scheduler are the ones the 3-step fit ended with (other: {diff})")
    check(s6.get("opt") != s.get("opt") and trainer6.global_step == 5 and len(COPIES) == 1, "f: the optimizer state is no longer a fresh run's; the fit ran to step 5")
    os.remove(ckpt)


def main(scratch):
    os.makedirs(scratch, exist_ok=True)
    part_a(scratch)
    part_bcdf(scratch)
    print(f"TEST_LAUNCH_FINGERPRINT {'FAIL' if BAD else 'PASS'}: {len(BAD)} failures")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
