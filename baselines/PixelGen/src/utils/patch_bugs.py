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

    _SIGTERM_EVERY = max(1, int(os.environ.get("GPIC_SIGTERM_EVERY") or 50))   # steps; one check costs about 30 ms at d640, 55 ms at d1280 (w1048)
    if os.environ.get("GPIC_SIGTERM_EVERY"):
        print(f"[sigterm] broadcast every {_SIGTERM_EVERY} steps", flush=True)

    def _broadcast_sigterm_sometimes(self):
        if self.trainer.global_step % _SIGTERM_EVERY == 0:
            _broadcast_sigterm_every_step(self)

    _TEL._broadcast_sigterm_tensor = _broadcast_sigterm_sometimes

# Crash fix (genmodel 2026-11-07: Python aborted in none_dealloc in w1204 after 139 min and in w1239, both with
# TORCHINDUCTOR_CPP_WRAPPER=1, the main thread inside the training step). torch 2.7.1's C++ wrapper
# (torch/_inductor/codecache.py, CppWrapperCodeCache.extra_parse_arg, pack_tensor_handle_list) puts Py_None into the
# list of a graph's outputs with PyList_SET_ITEM, which steals a reference that was never taken: every None output of
# every compiled call (a backward graph returns None for each input without a gradient) drops None's refcount by one,
# and in Python 3.11 None is not immortal, so at zero the interpreter aborts. Take the reference (Py_NewRef). The
# template joins the C++ source when a wrapper is loaded (load_pybinding_async), also from the FX graph cache, so the
# fixed source gets its own hash and compiles once; nothing else changes (tools/mfu/tests/test_none_refcount.py).
try:
    from torch._inductor.codecache import CppWrapperCodeCache as _CWC
    if "? Py_None" in _CWC.extra_parse_arg:
        _CWC.extra_parse_arg = _CWC.extra_parse_arg.replace("? Py_None", "? Py_NewRef(Py_None)")
except Exception as _e:   # a newer torch without the template: nothing to fix here
    print(f"[patch] the Py_None fix of the C++ wrapper was not applied: {_e!r}", flush=True)
