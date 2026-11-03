"""ladder: the host side of a training run, with the math unchanged (docs/text_cache.md, jobs/smoke_ladder_host.sh).
1. Each rank and its data workers stay on the CPUs of its GPU's socket, so kernel launches and pinned copies stay local.
2. After the warm-up, gc.freeze: a full collection no longer walks the compiled graphs' objects. Before, one took about 0.5 s on
   one rank, and the other seven waited for it.
Place it after CompileWarmup in the callback list.
"""
import gc
import os
import subprocess

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


def pin_rank_threads(index, workers):
    """The rank's main thread and its autograd thread alternate, so they share one CPU that no data worker uses: no wait for a CPU
    and no worker in their caches. Each socket gives up one CPU per GPU on it (index % 4: GPUs 0-3 on one socket, 4-7 on the other).
    The pin thread and the data workers (workers: their pids) stay on the other CPUs of the socket. Call it on the main thread,
    after the first backward (the autograd thread exists) and after the workers started (later children would inherit one CPU)."""
    socket = sorted(os.sched_getaffinity(0))
    if not 8 <= len(socket) <= 64:          # outside this range bind_to_gpu_socket did not run: two ranks would get one CPU
        raise ValueError(f"{len(socket)} CPUs for this rank")
    reserved = socket[-4:]                  # the last CPUs: the first CPU of a socket gets more of the system's interrupts
    core = reserved[index % 4]
    others = set(socket) - set(reserved)
    threads = {}
    for tid in os.listdir("/proc/self/task"):
        with open(f"/proc/self/task/{tid}/comm") as f:
            threads[int(tid)] = f.read().strip()
    autograd = [t for t, name in threads.items() if name == f"pt_autograd_{index}"]
    if not autograd:
        raise ValueError(f"no autograd thread among {sorted(set(threads.values()))}")
    for pid in workers:
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                os.sched_setaffinity(int(tid), others)
        except OSError:                     # a worker that ended, or a thread that just ended
            pass
    for t, name in threads.items():
        if name == "pt_data_pin":
            os.sched_setaffinity(t, others)
    for t in autograd:
        os.sched_setaffinity(t, {core})
    os.sched_setaffinity(0, {core})         # this thread
    print(f"[numa] device {index}: main and autograd threads on CPU {core}, {len(workers)} workers on {len(others)} CPUs", flush=True)


class HostTuning(Callback):
    def __init__(self):
        super().__init__()
        self._pin = os.environ.get("GPIC_PROF_PIN") == "1"   # profiling switch (jobs/smoke_ladder_kernels.sh)

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

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._pin:
            self._pin = False
            try:
                it = getattr(getattr(trainer.datamodule.train_dataset, "_dataloader", None), "_iterator", None)
                pin_rank_threads(trainer.strategy.root_device.index, [w.pid for w in getattr(it, "_workers", [])])
            except Exception as e:
                print(f"[numa] pin skipped: {e!r}", flush=True)
