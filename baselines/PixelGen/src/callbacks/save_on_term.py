"""Save on the pilot's cut (gpic-28 2026-11-09; owner: 「修掉」 on genmodel's bug list).

fetch_pending runs a wq guest under `timeout --kill-after=300 <s>`: at the planned end, timeout sends SIGTERM to the guest's whole
process group and SIGKILL 300 s later. Before this callback Lightning turned the TERM into a stop without a save, so a --partial
run lost every step since its last scheduled last.ckpt (up to max_steps / 8 in the sweeps: 20-60 min of 8 GPUs per cut).
Here the TERM only sets a flag on each rank. Every `every` optimizer steps the ranks take the MAX of their flags with one
all-reduce (one sync per `every` steps), so they all stop at the same step; then they save the full state through the trainer's
own path (trainer.save_checkpoint: the strategy's state dict, ShardClock's shard-log cut, the async writer) as last-term.ckpt in
the run directory (jobs/train_ladder.sh resumes from the newest last*.ckpt), wait for the write, and stop. The data workers
ignore TERM (gpic_dataset.build_dataloader), or they would die at once and fail the next batch before the save.
"""
import os
import signal
import time

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


class SaveOnTerm(Callback):
    def __init__(self, dirpath: str = None, every: int = 20, filename: str = "last-term.ckpt"):
        super().__init__()
        self.dirpath, self.filename = dirpath, filename
        self.every = int(os.environ.get("GPIC_TERM_CHECK_EVERY", every))
        self.flag, self.t_flag, self.done = False, None, False

    def on_train_start(self, trainer, pl_module):
        # after Lightning's own signal handlers (registered when the run starts): this one replaces its SIGTERM handler
        def handler(signum, frame):
            if not self.flag:
                self.flag, self.t_flag = True, time.time()
        signal.signal(signal.SIGTERM, handler)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.done or (batch_idx + 1) % max(1, trainer.accumulate_grad_batches) or trainer.global_step % self.every:
            return
        t = torch.tensor([1.0 if self.flag else 0.0], device=pl_module.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if t.item() == 0.0:
            return
        self.done = True
        d = self.dirpath or next((c.dirpath for c in trainer.checkpoint_callbacks if getattr(c, "dirpath", None)), None)
        path = os.path.join(d, self.filename)
        t0 = time.time()
        trainer.save_checkpoint(path)
        io = trainer.strategy.checkpoint_io
        if hasattr(io, "_reap"):   # AsyncShmCheckpointIO: wait until the write is on disk (rank 0 holds it)
            while any(b[1] is not None for b in getattr(io, "_bufs", [])):
                io._reap(block=True)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            waited = f"{t0 - self.t_flag:.1f} s after the TERM" if self.t_flag else "TERM on another rank"
            print(f"[save_on_term] step {trainer.global_step}: saved {path} in {time.time() - t0:.1f} s ({waited}); stopping", flush=True)
        trainer.should_stop = True
