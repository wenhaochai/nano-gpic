import torch
import lightning.pytorch.loggers.wandb as wandb

setattr(wandb, '_WANDB_AVAILABLE', True)
torch.set_float32_matmul_precision('medium')


import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)

import os
os.environ.setdefault("NCCL_DEBUG", "WARN")   # keep a value set by the job (NCCL_DEBUG=INFO for tuning logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# perf C7: Lightning 2.6 broadcasts a SIGTERM flag before every training batch (_TrainingEpochLoop.advance): a tensor
# built from a Python int (a pageable host-to-device copy), an NCCL broadcast and .item(), so two host syncs and a
# wait for every rank on every step (12-16 ms of host wait, then an idle GPU while the host catches up; 2026-11-01
# profile). Check before every 50th step instead: all ranks call it at the same global steps, so the collective stays
# matched; a SIGTERM is acted on within 50 steps.
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop as _TEL
if hasattr(_TEL, "_broadcast_sigterm_tensor"):
    _broadcast_sigterm_every_step = _TEL._broadcast_sigterm_tensor

    def _broadcast_sigterm_every_50(self):
        if self.trainer.global_step % 50 == 0:
            _broadcast_sigterm_every_step(self)

    _TEL._broadcast_sigterm_tensor = _broadcast_sigterm_every_50
