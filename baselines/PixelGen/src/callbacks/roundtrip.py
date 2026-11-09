"""In-process checkpoint round trip (gpic-28 2026-11-09; owner popup 「A 和 B 都做」, experiment B).

The FP8 resume test: a resume at 11,904 in a new process gave FD 851.0 (and 863.8 with the bucket fix) against 808.3 for
the same run continued, with equal held-out loss. A resume is two things: the checkpoint round trip (save, then load into
the model, optimizer and LR schedulers) and a new process (data workers, bucket pools, RNG streams, compiled graphs, FP8
history). At each step in GPIC_ROUNDTRIP_STEPS this callback saves a full-state checkpoint through the trainer's own path
(trainer.save_checkpoint: the strategy's full state dict, the async writer), waits for the write, and loads it back into
the running objects through the resume path (ModelParallelStrategy.load_checkpoint, then the LR schedulers' states); the
process and everything not in the checkpoint go on unchanged. If such a run scores like the continuous one, the round trip
is clean and the cost is in the new process; if it scores like the resumes, the round trip is where to look.
"""
import os
import time

from lightning.pytorch.callbacks import Callback


class RoundTrip(Callback):
    def __init__(self, steps=None, dirpath=None):
        super().__init__()
        env = os.environ.get("GPIC_ROUNDTRIP_STEPS", "")
        self.steps = sorted(int(s) for s in (steps or env.split(",")) if str(s).strip())
        self.dirpath, self.done = dirpath, set()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # at the start of the batch after the optimizer step: the state is that of global step N, all checkpoints of the
        # step before it are written (ModelCheckpoint callbacks run last in on_train_batch_end)
        n = trainer.global_step
        if n not in self.steps or n in self.done:
            return
        self.done.add(n)
        d = self.dirpath or next((c.dirpath for c in trainer.checkpoint_callbacks if getattr(c, "dirpath", None)), None)
        path = os.path.join(d, f"roundtrip-step={n:06d}.ckpt")
        t0 = time.time()
        trainer.save_checkpoint(path)
        io = trainer.strategy.checkpoint_io
        if hasattr(io, "_reap"):   # AsyncShmCheckpointIO: wait until every pending write is on disk
            while any(b[1] is not None for b in getattr(io, "_bufs", [])):
                io._reap(block=True)
        trainer.strategy.barrier()
        t1 = time.time()
        ck = trainer.strategy.load_checkpoint(path)
        for cfg, s in zip(trainer.lr_scheduler_configs, ck.get("lr_schedulers", [])):
            cfg.scheduler.load_state_dict(s)
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            lrs = [round(g["lr"], 8) for g in trainer.optimizers[0].param_groups]
            print(f"[roundtrip] step {n}: saved {path} in {t1 - t0:.1f} s, loaded it back in {time.time() - t1:.1f} s "
                  f"(checkpoint global_step {ck.get('global_step')}, lr {lrs})", flush=True)
