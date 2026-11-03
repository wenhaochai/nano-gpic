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


class HostTuning(Callback):
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
