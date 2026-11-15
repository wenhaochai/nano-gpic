"""tests only (jobs/smoke_launch.sh; GPIC_LAUNCH_FP=1 in jobs/train_ladder.sh): the state a launch starts its first step
from, and when it gets there, one line per rank and event, so two launches that differ only in the launch levers can be
compared field by field (tools/mfu/check_launch.py). It draws no random number and writes to no tensor of the run; its
sums use temporary memory on the tensor's device. Place it last in the callback list.

[launch-fp] rank r start ...    at the end of on_train_start: hashes of the parameters and buffers of the denoiser and of
                                the EMA slot (this rank's shard under FSDP2), of the optimizer state and of the four
                                random streams; the compiler's vector sets and the memo's counts (src/utils/patch_bugs.py);
                                thread counts now (threads = intra-op/inter-op, omp = OMP_NUM_THREADS) and when this
                                module was imported (thr0, omp0: before fit, where Lightning sets them);
                                seconds from GPIC_LAUNCH_T0 (the start of train_ladder.sh) to the process
                                start, this callback's construction, setup, on_fit_start and on_train_start; the
                                seconds and the minor page faults of the EMA slot's copy in configure_model
[launch-fp] rank r batch i ...  the batch's hash and the seconds at on_train_batch_start, for the first two batches of
                                this launch (i counts from 0 in every launch, also in a resumed one)
Each line goes out in one write with its newline, so the lines of 8 ranks in one log stay whole.
"""
import hashlib
import os
import pickle
import random
import resource
import sys
import time

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

_T0 = float(os.environ.get("GPIC_LAUNCH_T0") or time.time())
_THR0, _OMP0 = torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS", "unset")   # the CLI imports this module while it parses
_M64 = (1 << 64) - 1
_INT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _proc_start():
    """Seconds from T0 to this process's start (/proc: the start in clock ticks after boot)."""
    try:
        with open("/proc/self/stat") as f:
            ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        return time.time() - (time.clock_gettime(time.CLOCK_BOOTTIME) - ticks / os.sysconf("SC_CLK_TCK")) - _T0
    except Exception:
        return float("nan")


@torch.no_grad()
def tensor_sums(t):
    """(shape, dtype, sum, position-weighted sum) of the tensor's bits read as integers, modulo 2^64: equal tensors give
    equal values, and a changed, moved or swapped element changes them. A DTensor gives its local shard."""
    if hasattr(t, "to_local"):
        t = t.to_local()
    t = t.detach()
    v = torch.view_as_real(t) if t.is_complex() else t
    v = v.contiguous().reshape(-1)
    if v.dtype == torch.bool:
        v = v.to(torch.uint8)
    elif v.dtype.is_floating_point:
        v = v.view(_INT[v.element_size()])
    s = ws = 0
    for i in range(0, v.numel(), 1 << 24):   # 16 M elements at a time: at most 0.4 GiB of temporaries
        c = v[i:i + (1 << 24)].to(torch.int64)
        s += int(c.sum().item())
        ws += int((c * torch.arange(i + 1, i + 1 + c.numel(), device=c.device, dtype=torch.int64)).sum().item())
    return tuple(t.shape), str(t.dtype), s & _M64, ws & _M64


def walk(o, out):
    """The tensors, numbers and strings of a nested batch or optimizer state, in order. Tensor keys (an optimizer's
    state is keyed by its parameters) are skipped; their values keep the order."""
    if torch.is_tensor(o):
        out.append(tensor_sums(o))
    elif isinstance(o, dict):
        for k, v in o.items():
            if not torch.is_tensor(k):
                out.append(repr(k))
            walk(v, out)
    elif isinstance(o, (list, tuple)):
        out.append(len(o))
        for v in o:
            walk(v, out)
    elif isinstance(o, str):
        out.append(hashlib.sha1(o.encode()).hexdigest()[:8])
    elif isinstance(o, (bool, int, float, type(None))):
        out.append(repr(o))
    elif isinstance(o, np.ndarray):
        out.append(hashlib.sha1(o.tobytes()).hexdigest()[:8])
    else:
        out.append(type(o).__name__)
    return out


def say(line):
    sys.stdout.write(f"[launch-fp] {line}\n")   # one write: PYTHONUNBUFFERED splits print() into the text and the newline
    sys.stdout.flush()


def digest(items):
    h = hashlib.sha1()
    for x in items:
        h.update(repr(x).encode())
        h.update(b"\n")
    return f"{h.hexdigest()[:12]}/{len(items)}"


def state_fields(denoiser, ema_denoiser, optimizers=(), schedulers=(), device=None):
    """The hashes of one rank's state: a dict of short strings."""
    f = {}
    for tag, m in (("den", denoiser), ("ema", ema_denoiser)):
        f[tag] = digest([(n, tensor_sums(p)) for n, p in m.named_parameters()])
        f[tag + "_buf"] = digest([(n, tensor_sums(b)) for n, b in m.named_buffers()])
    f["opt"] = digest(walk([[o.state, [{k: v for k, v in g.items() if k != "params"} for g in o.param_groups]]
                            for o in optimizers], []))
    f["sched"] = digest(walk([s.state_dict() for s in schedulers], []))
    f["rng_cpu"] = hashlib.sha1(torch.get_rng_state().numpy().tobytes()).hexdigest()[:12]
    if device is not None and torch.device(device).type == "cuda":
        f["rng_cuda"] = hashlib.sha1(torch.cuda.get_rng_state(device).numpy().tobytes()).hexdigest()[:12]
    f["rng_np"] = hashlib.sha1(pickle.dumps(np.random.get_state())).hexdigest()[:12]
    f["rng_py"] = hashlib.sha1(repr(random.getstate()).encode()).hexdigest()[:12]
    return f


class LaunchFingerprint(Callback):
    def __init__(self):
        super().__init__()
        self.t = {"proc": _proc_start(), "init": time.time() - _T0}
        self.seen = 0
        self.copy = None

    def setup(self, trainer, pl_module, stage):
        self.t["setup"] = time.time() - _T0
        # configure_model copies the new weights into the EMA slot, which is a deepcopy of them already (4.2 s at width
        # 1536 in the w2441 profile): the call is timed here, through the module's own name for it, and not changed
        mod = sys.modules.get(type(pl_module).__module__)
        inner = getattr(mod, "copy_params", None)
        if callable(inner) and not hasattr(inner, "_fp_inner"):
            def timed(*a, **k):
                t0, f0 = time.time(), resource.getrusage(resource.RUSAGE_SELF).ru_minflt
                try:
                    return inner(*a, **k)
                finally:
                    mod.copy_params = inner   # the first call only: configure_model's
                    self.copy = (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0)
            timed._fp_inner = inner
            mod.copy_params = timed

    def on_fit_start(self, trainer, pl_module):
        self.t["fit"] = time.time() - _T0

    def on_train_start(self, trainer, pl_module):
        self.t["train"] = time.time() - _T0
        t1 = time.time()
        f = {"step": trainer.global_step}
        f.update(state_fields(pl_module.denoiser, pl_module.ema_denoiser, trainer.optimizers,
                              [c.scheduler for c in trainer.lr_scheduler_configs], pl_module.device))
        try:   # cached per process: the compiler asked for them in the warm-up
            from torch._inductor.cpu_vec_isa import valid_vec_isa_list
            f["isa"] = ",".join(type(i).__name__ for i in valid_vec_isa_list()) or "none"
        except Exception as e:
            f["isa"] = f"error:{type(e).__name__}"
        memo = getattr(sys.modules.get("src.utils.patch_bugs"), "VEC_ISA_MEMO_STATS", None)
        f["memo"] = "/".join(f"{k[0]}{v}" for k, v in memo.items()) if memo else "none"
        f["threads"] = f"{torch.get_num_threads()}/{torch.get_num_interop_threads()}"
        f["omp"] = os.environ.get("OMP_NUM_THREADS", "unset")
        f["thr0"], f["omp0"] = _THR0, _OMP0
        f["cpus"] = len(os.sched_getaffinity(0))
        f.update({"t_" + k: f"{v:.2f}" for k, v in self.t.items()})
        f["copy_s"], f["copy_flt"] = (f"{self.copy[0]:.2f}", self.copy[1]) if self.copy else ("none", "none")
        f["fp_s"] = f"{time.time() - t1:.2f}"
        say(f"rank {trainer.global_rank} start " + " ".join(f"{k}={v}" for k, v in f.items()))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.seen < 2:   # batch_idx goes on from the checkpoint's count in a resumed launch
            t = time.time() - _T0   # before the hash
            say(f"rank {trainer.global_rank} batch {self.seen} step={trainer.global_step} hash={digest(walk(batch, []))} t={t:.2f}")
            self.seen += 1
