import torch
import lightning.pytorch.loggers.wandb as wandb

setattr(wandb, '_WANDB_AVAILABLE', True)
torch.set_float32_matmul_precision('medium')

# perf C5: cuDNN attention first. At this job's shapes it beats flash on H100 (microbench, 2026-11-01:
# [32,24,256->556,64] fwd+bwd 0.59 vs 0.83 ms; [32,24,300,300,64] 0.50 vs 0.76 ms). Set before any torch.compile so the
# traced graphs pick it; other backends stay as fallbacks in their usual order.
from torch.nn.attention import SDPBackend as _B
torch.backends.cuda.enable_cudnn_sdp(True)
torch._C._set_sdp_priority_order([int(_B.CUDNN_ATTENTION), int(_B.FLASH_ATTENTION), int(_B.EFFICIENT_ATTENTION),
                                  int(_B.MATH)] + [int(b) for b in torch._C._get_sdp_priority_order()
                                                   if int(b) not in (int(_B.CUDNN_ATTENTION), int(_B.FLASH_ATTENTION),
                                                                     int(_B.EFFICIENT_ATTENTION), int(_B.MATH))])

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)

import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

