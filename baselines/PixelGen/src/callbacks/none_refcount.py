"""Probe for "Fatal Python error: none_dealloc" (w1239, w1204): a C extension that releases a reference to None it never
took lowers None's refcount by one per call until it reaches zero. This callback logs sys.getrefcount(None) on rank 0 after
every training micro-batch, so the drop per micro-batch and per optimizer step shows the leak and its rate."""
import sys
import time

import lightning.pytorch as pl


class NoneRefcount(pl.Callback):
    def __init__(self, path: str, every: int = 1):
        self.path, self.every = path, every

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0 and batch_idx % self.every == 0:
            with open(self.path, "a") as f:
                f.write(f"{time.time():.3f} batch {batch_idx} step {trainer.global_step} none_refs {sys.getrefcount(None)}\n")
