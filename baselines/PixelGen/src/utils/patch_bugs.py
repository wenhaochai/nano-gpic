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

# Launch lever E (gpic-28 2026-11-09). Before its first compile every rank asks torch._inductor.cpu_vec_isa which CPU
# vector sets it may use (valid_vec_isa_list: AMX, AVX-512, AVX2). Each answer is VecISA.check_build: under a file lock
# it starts a new Python that imports torch and loads a small test library, four times per rank and 8 ranks at once,
# about 13 s of the warm-up of every launch (py-spy, w2441; the library itself is never used again). The answer depends
# only on the compiler, the flags, the torch build, the interpreter and the CPU, so a True answer is kept as an empty
# file named by a hash of all of them in $GPIC_VEC_ISA_MEMO (set by jobs/train_ladder.sh), and a later launch reads
# it. A False answer is never stored, and any error here falls back to the real check, so the list of vector sets
# (part of the compile cache key) is the one the real check gives. TORCHINDUCTOR_VEC_ISA_OK=1 is not used: it enters
# the cache key and keeps the AMX check.
VEC_ISA_MEMO_STATS = {"hit": 0, "miss": 0, "stored": 0, "error": 0}
_VEC_ISA_MEMO = os.environ.get("GPIC_VEC_ISA_MEMO")
if _VEC_ISA_MEMO:
    try:
        import hashlib
        import platform
        import sys
        from torch._inductor import cpu_vec_isa as _cvi

        _check_build_real = _cvi.VecISA.check_build

        def _vec_isa_cpu():
            model = flags = ""
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if not model and line.startswith("model name"):
                        model = line
                    elif not flags and line.startswith("flags"):
                        flags = line
                    if model and flags:
                        break
            if not (model and flags):
                raise ValueError("no CPU model or flags in /proc/cpuinfo")
            return model + flags

        def _vec_isa_memo_path(isa, code):
            parts = [code, _cvi._get_isa_dry_compile_fingerprint(isa._arch_flags), type(isa).__name__, str(isa),
                     sys.executable, torch.__file__, torch.__version__, str(torch.version.git_version),
                     platform.machine(), platform.release(), _vec_isa_cpu()]
            return os.path.join(_VEC_ISA_MEMO, hashlib.sha256("\0".join(parts).encode()).hexdigest())

        def _check_build_memo(self, code):
            try:
                path = _vec_isa_memo_path(self, code)
                if os.path.isfile(path):
                    VEC_ISA_MEMO_STATS["hit"] += 1
                    return True
            except Exception:
                VEC_ISA_MEMO_STATS["error"] += 1
                return _check_build_real(self, code)
            VEC_ISA_MEMO_STATS["miss"] += 1
            ok = _check_build_real(self, code)
            if ok is True:
                try:
                    os.makedirs(_VEC_ISA_MEMO, exist_ok=True)
                    tmp = f"{path}.{os.uname().nodename}.{os.getpid()}.tmp"
                    with open(tmp, "w"):
                        pass
                    os.replace(tmp, path)
                    VEC_ISA_MEMO_STATS["stored"] += 1
                except Exception:
                    VEC_ISA_MEMO_STATS["error"] += 1
            return ok

        _cvi.VecISA.check_build = _check_build_memo
    except Exception as _e:   # a torch without this check: the launch runs as before
        print(f"[patch] the vector-set memo is off: {_e!r}", flush=True)
