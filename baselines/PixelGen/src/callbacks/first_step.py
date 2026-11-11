"""The first step of a launch (2026-11-10, w2503): jobs/train_ladder.sh retries a launch that failed before its first step
and ends one that has not reached it GPIC_LAUNCH_START_TIMEOUT_S seconds after its start. It learns that the first step has
run from this file: rank 0 writes GPIC_FIRST_STEP_FILE (one line: the step and the time) at the end of the first training
batch of this process (the first micro-batch under accumulation: training has begun). The script names the file after the
launch (GPIC_LAUNCH_ID), removes it before and after the launch, so it is per launch, also on a resume, where global_step
starts at the checkpoint's step and no file of an earlier launch can count. A write that fails is tried again at the next
batch (without the file the watchdog would end a healthy launch). No file name in the environment: nothing happens."""
import os
import time

from lightning.pytorch.callbacks import Callback


class FirstStepMarker(Callback):
    def __init__(self):
        super().__init__()
        self.path = os.environ.get("GPIC_FIRST_STEP_FILE") or None
        self.warned = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.path is None or not trainer.is_global_zero:
            return
        try:
            with open(self.path + ".tmp", "w") as f:
                f.write(f"{trainer.global_step} {time.time():.3f}\n")
            os.replace(self.path + ".tmp", self.path)
            self.path = None   # once
        except OSError as e:   # the training must not fail for the marker: again at the next batch
            if not self.warned:
                print(f"[first_step] cannot write {self.path}: {e!r}; trying again at the next batch", flush=True)
                self.warned = True
