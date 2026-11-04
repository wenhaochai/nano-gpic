"""ladder: the host side of a training run, with the math unchanged (docs/text_cache.md, jobs/smoke_ladder_host.sh).
1. Each rank and its data workers stay on the CPUs of its GPU's socket, so kernel launches and pinned copies stay local.
2. After the warm-up, gc.freeze: a full collection no longer walks the compiled graphs' objects. Before, one took about 0.5 s on
   one rank, and the other seven waited for it.
Place it after CompileWarmup in the callback list.
"""
import gc
import os
import subprocess
import threading

import torch
from lightning.pytorch.callbacks import Callback


def bind_to_gpu_socket(index):
    """This thread, its later threads and children on the CPUs of the GPU's NUMA node."""
    uuid = f"GPU-{torch.cuda.get_device_properties(index).uuid}"
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True)
    bus = dict(line.split(", ") for line in out.strip().splitlines())[uuid].lower()[4:]   # 00000000:e1:00.0 -> 0000:e1:00.0
    with open(f"/sys/bus/pci/devices/{bus}/numa_node") as f:
        node = int(f.read())
    if node < 0:
        raise ValueError(f"no NUMA node for {bus}")
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        cpus = set()
        for part in f.read().strip().split(","):
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
    allowed = cpus & os.sched_getaffinity(0)
    if allowed:
        os.sched_setaffinity(0, allowed)
    print(f"[numa] device {index} {bus} node {node}: {len(os.sched_getaffinity(0))} CPUs", flush=True)


def pin_rank_threads(index, workers, split=False):
    """The rank's main thread and its autograd thread alternate, so they share one CPU that no other thread of the node uses: no
    wait for a CPU and nothing else in their caches. Each socket gives up one CPU per GPU on it (index % 4: GPUs 0-3 on one socket,
    4-7 on the other). Every other thread of the rank (pin, NCCL, CUDA, idle autograd threads) and the data workers (workers: their
    pids) stay on the other CPUs of the socket. Call it on the main thread, after the first backward (the autograd thread exists)
    and after the workers started (later children would inherit one CPU).
    split: main and autograd each get their own CPU, so each socket gives up two CPUs per GPU. w1048 stalled with one shared CPU: a run queue
    with two pinned threads can be the busiest queue, and a balance pass that can pull neither doubles its interval (kernel: LBF_ALL_PINNED), so
    the workers may wait for a CPU while others are idle. A queue with one running thread is not a pull candidate. jobs/smoke_ladder_pin2.sh tests this."""
    socket = sorted(os.sched_getaffinity(0))
    if not 16 <= len(socket) <= 64:         # outside this range bind_to_gpu_socket did not run: two ranks would get one CPU
        raise ValueError(f"{len(socket)} CPUs for this rank")
    reserved = socket[-8:] if split else socket[-4:]   # the last CPUs: the first CPU of a socket gets more of the system's interrupts
    core = reserved[index % 4]
    autograd_core = reserved[4 + index % 4] if split else core
    others = set(socket) - set(reserved)
    threads = {}
    for tid in os.listdir("/proc/self/task"):
        with open(f"/proc/self/task/{tid}/comm") as f:
            threads[int(tid)] = f.read().strip()
    autograd = [t for t, name in threads.items() if name == f"pt_autograd_{index}"]
    if not autograd:
        raise ValueError(f"no autograd thread among {sorted(set(threads.values()))}")
    moved = 0
    for pid in workers:
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                os.sched_setaffinity(int(tid), others)
            moved += 1
        except OSError:                     # a worker that ended, or a thread that just ended
            pass
    me = threading.get_native_id()
    for t in threads:
        if t != me and t not in autograd:
            try:
                os.sched_setaffinity(t, others)
            except OSError:                 # a thread that just ended
                pass
    for t in autograd:
        os.sched_setaffinity(t, {autograd_core})
    os.sched_setaffinity(0, {core})         # this thread
    print(f"[numa] device {index}: main thread on CPU {core}, autograd thread on CPU {autograd_core}, {moved} workers and {len(threads) - 1 - len(autograd)} other threads on {len(others)} CPUs", flush=True)


def idle_workers(index, workers):
    """Data workers under SCHED_IDLE, on all their CPUs. A waking main or autograd thread then preempts a worker at once, and a CPU that runs only
    workers counts as idle when the scheduler places it (kernel: check_preempt_wakeup, sched_idle_cpu), so those threads rarely wait for a CPU (w1048:
    8.5% and 7.7% of the wall clock), and the workers lose no CPU, since nothing else wants one. A nice value cannot do this here: jobs run at nice 19,
    so all threads already have the highest value (w1051: "0 of 32 workers at nice 10"). Moving to SCHED_IDLE needs no privilege."""
    done = 0
    for pid in workers:
        try:
            tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
        except OSError:                     # a worker that ended
            continue
        for tid in tids:
            try:
                os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
                done += tid == pid
            except OSError:                 # a thread that just ended
                pass
    print(f"[idle] device {index}: {done} of {len(workers)} workers under SCHED_IDLE", flush=True)


class HostTuning(Callback):
    def __init__(self):
        super().__init__()
        self._pin = os.environ.get("GPIC_PROF_PIN") in ("1", "2")   # profiling switch: 1 one shared CPU (jobs/smoke_ladder_kernels.sh), 2 one CPU each (jobs/smoke_ladder_pin2.sh)
        self._pin_split = os.environ.get("GPIC_PROF_PIN") == "2"
        self._idle = os.environ.get("GPIC_WORKER_IDLE") == "1"   # jobs/smoke_ladder_host3.sh

    def setup(self, trainer, pl_module, stage):
        # before the dataloaders start, so the data workers follow the rank; a few percent of speed never stops a run
        if stage == "fit" and trainer.strategy.root_device.type == "cuda":
            try:
                bind_to_gpu_socket(trainer.strategy.root_device.index)
            except Exception as e:
                print(f"[numa] skipped: {e!r}", flush=True)

    def on_train_start(self, trainer, pl_module):
        gc.collect()
        gc.freeze()
        if os.environ.get("GPIC_RANK_SEED") == "1":
            # Lightning seeds every rank alike (reset_seed), so without this all ranks draw the same t, noise and CFG dropout at each step
            # and a global batch of 256 has 32 different (t, noise). Applied after the warm-up. jobs/smoke_ladder_seed.sh
            seed = torch.initial_seed() % 2**31 + 1000003 * (trainer.global_rank + 1)
            torch.manual_seed(seed)
            print(f"[seed] rank {trainer.global_rank}: {seed}", flush=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._idle:
            self._idle = False
            try:
                it = getattr(getattr(trainer.datamodule.train_dataset, "_dataloader", None), "_iterator", None)
                idle_workers(trainer.strategy.root_device.index, [w.pid for w in getattr(it, "_workers", []) if w.is_alive()])
            except Exception as e:
                print(f"[idle] skipped: {e!r}", flush=True)
        if self._pin:
            self._pin = False
            try:
                it = getattr(getattr(trainer.datamodule.train_dataset, "_dataloader", None), "_iterator", None)
                # is_alive: the list keeps workers that ended (GPIC_PROF_CYCLE_BATCHES), and their pid numbers can come back
                pin_rank_threads(trainer.strategy.root_device.index, [w.pid for w in getattr(it, "_workers", []) if w.is_alive()], split=self._pin_split)
            except Exception as e:
                print(f"[numa] pin skipped: {e!r}", flush=True)
