"""Monitor for "Fatal Python error: none_dealloc" (w1239, w1204): a C extension that releases a reference to None it never
took lowers None's refcount until it reaches zero (Python 3.11: None is not immortal). The cause found (torch 2.7.1's C++
wrapper, pack_tensor_handle_list) is fixed in src/utils/patch_bugs.py; this callback logs sys.getrefcount(None) on rank 0
at the run's first step and then every `every` optimizer steps, so a run shows its margin and any new leak's rate."""
import os
import sys
import time

import lightning.pytorch as pl


class NoneRefcount(pl.Callback):
    def __init__(self, path: str = "", every: int = 1000):
        self.path, self.every, self._last = path, every, None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if trainer.global_rank != 0 or step == self._last or (step % self.every and self._last is not None):
            return
        self._last = step
        path = self.path or os.path.join(trainer.default_root_dir, "none_refcount.txt")
        with open(path, "a") as f:
            f.write(f"{time.time():.3f} batch {batch_idx} step {step} none_refs {sys.getrefcount(None)}\n")
