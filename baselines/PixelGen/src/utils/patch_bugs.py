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

