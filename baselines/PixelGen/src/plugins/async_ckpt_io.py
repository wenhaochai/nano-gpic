"""MFU lever F (docs/mfu.md): checkpoint writes off the training path, the same files.

Lightning saves on rank 0 (torch.save into memory, then a write to GPFS) while every rank waits at the barrier after it:
57.8 s per last.ckpt at d1792 (27 GB), 36.6 s at d1536, 8 saves per C0, about 6% and 4% of those runs. Here rank 0 only
copies the checkpoint's tensors into a shared-memory buffer (page-locked when CUDA allows it, so the device-to-host copy
runs at full speed) and hands the structure to a writer process, which rebuilds each tensor, torch.saves the dict to a
temporary file and renames it into place (atomic, as before). Two buffers, so the ms checkpoint and last.ckpt of the same
step can both be in flight; a third save waits for a buffer, and teardown waits for every write. Lightning 2.6.6 calls
teardown from DDP's strategy but not from ModelParallelStrategy (lever B), whose teardown skips checkpoint_io.teardown, so
the writes are also awaited at interpreter exit (atexit runs before multiprocessing ends its daemon children; without it,
the final checkpoints of an FSDP2 run were lost whenever their write outlasted the exit). The file's mtime is set
to the snapshot time, because jobs/train_ladder.sh keeps the shard-log lines up to a resumed checkpoint's mtime. The file
is what torch.save writes from the same dict: the values are copies, no math changes.
Twins (2026-11-09): under GPIC_MFU_NO_EMA the EMA slot of every checkpoint holds the raw weights (SimpleEMA.mirror), so each
file held the weights twice (d1536: ms 8.98 GB, last.ckpt 13.5 GB; 4.85 TiB of checkpoints in 48 h). A state_dict tensor
ema_denoiser.X whose bytes equal denoiser.X's (compared here at every save, bit for bit) is not copied again: both keys get
the same bytes of the buffer and the writer builds both tensors on one storage, which torch.save writes once. torch.load
gives two tensors with the values the old file gave (they share a storage, as any two views do); load_state_dict copies
them into the modules as before. ms files halve, last.ckpt loses a third; with an EMA that moves, nothing is equal and
the file is the old one.
Use: trainer.plugins += [src.plugins.async_ckpt_io.AsyncShmCheckpointIO] (GPIC_MFU_ASYNC_CKPT=1 in jobs/mfu/arm.sh).
"""
import atexit
import glob
import os
import queue
import re
import signal
import sys
import time

import torch
import torch.multiprocessing as mp
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.apply_func import apply_to_collection


class _Ref:
    """A tensor in the shared buffer: byte offset, dtype, shape (picklable)."""

    def __init__(self, off, dtype, shape):
        self.off, self.dtype, self.shape = off, dtype, tuple(shape)


_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}   # element size -> a dtype to compare bytes with


def _twins(checkpoint):
    """{id(t): r} for each state_dict tensor t = ema_denoiser.X with the bytes of r = denoiser.X (same dtype, shape and
    device; compared as integers, so a NaN equals itself and -0.0 differs from 0.0)."""
    sd = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    out = {}
    if not isinstance(sd, dict):
        return out
    for key, t in sd.items():
        if not (isinstance(key, str) and key.startswith("ema_denoiser.") and isinstance(t, torch.Tensor)):
            continue
        r = sd.get("denoiser." + key[len("ema_denoiser."):])
        if not isinstance(r, torch.Tensor) or r is t or (r.dtype, r.shape, r.device) != (t.dtype, t.shape, t.device):
            continue
        i = _INT.get(t.element_size())
        try:
            if i is not None and t.numel() and torch.equal(t.detach().view(i), r.detach().view(i)):
                out[id(t)] = r
        except Exception:   # a tensor that has no integer view (e.g. not strided): written on its own, as before
            pass
    return out


def _writer(jobs, acks):
    """The writer process: rebuild the tensors from the shared buffer, torch.save, rename, set the mtime."""
    # review S3: a SIGTERM to the job's process group must not cut a write short (it left a 13-27 GB *.tmp<pid>); the
    # training process waits for the writes at exit, and a SIGKILL leftover is removed at the next launch (_clean_tmp)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = os.getppid()
    while True:
        try:
            job = jobs.get(timeout=5)
        except queue.Empty:
            if os.getppid() != parent:   # the training process is gone: stop (SIGTERM no longer stops this process)
                return
            continue
        if job is None:
            return
        k, buf, snap, path, t_snap = job
        try:
            delay = float(os.environ.get("GPIC_ASYNC_CKPT_TEST_DELAY", "0"))   # tests only: a slow file system
            if delay:
                time.sleep(delay)
            st = buf.untyped_storage()
            parts = {}   # byte offset -> the storage there (tensors with bytes have distinct offsets; twins share one)

            def rebuild(r):
                # review M1: no copy (27 GB of host memory at d1792): each tensor gets its own storage, a slice of the
                # shared buffer, so torch.save writes the same bytes as for an independent copy. Twins (save_checkpoint
                # gave them one offset) are two tensors on one storage: torch.save writes a storage once.
                n = int(torch.Size(r.shape).numel()) * torch.empty((), dtype=r.dtype).element_size()
                if n == 0:   # an empty slice would start where the next tensor starts: torch.save refuses that
                    return torch.empty(r.shape, dtype=r.dtype)
                if r.off not in parts:
                    parts[r.off] = st[r.off:r.off + n]
                return torch.empty(0, dtype=r.dtype).set_(parts[r.off], 0, r.shape)
            ckpt = apply_to_collection(snap, _Ref, rebuild)
            tmp = f"{path}.tmp{os.getpid()}"
            torch.save(ckpt, tmp)
            os.replace(tmp, path)
            os.utime(path, (t_snap, t_snap))
            del ckpt
            parts.clear()
            acks.put((k, path, None))
        except Exception as e:   # raised in the training process at its next save or at teardown
            acks.put((k, path, repr(e)))
        del buf


def _clean_tmp(directory):
    """Temporary files of writers that are gone (review S3: a SIGKILL during a write leaves path.tmp<pid>)."""
    for f in glob.glob(os.path.join(directory, "*.tmp*")):
        m = re.search(r"\.tmp(\d+)$", f)
        if not m:
            continue
        try:
            os.kill(int(m.group(1)), 0)   # that writer still runs (another job on this node): leave its file
        except ProcessLookupError:
            try:
                if time.time() - os.path.getmtime(f) < 1800:   # a writer of another node's job could own a fresh one
                    continue
                os.remove(f)
                print(f"AsyncShmCheckpointIO: removed the leftover {f}", flush=True)
            except OSError:
                pass
        except PermissionError:
            pass


class AsyncShmCheckpointIO(TorchCheckpointIO):
    MAX_BUFS = 2

    def __init__(self):
        super().__init__()
        self._bufs = []      # [shared uint8 tensor, path being written or None]
        self._proc = None

    def _reap(self, block):
        while any(b[1] is not None for b in self._bufs):
            try:
                k, path, err = self._acks.get(block=block, timeout=10 if block else None)
            except queue.Empty:
                if block and not self._proc.is_alive():   # never wait forever on a writer that died (e.g. killed for memory)
                    raise RuntimeError(f"async checkpoint writer exited ({self._proc.exitcode}) with writes pending: "
                                       f"{[b[1] for b in self._bufs if b[1] is not None]}")
                if block:
                    continue
                return
            self._bufs[k][1] = None
            if err is not None:
                raise RuntimeError(f"async checkpoint write of {path} failed: {err}")
            if not block:
                continue
            return

    def _buffer(self, need):
        self._reap(block=False)
        while True:
            free = [k for k, b in enumerate(self._bufs) if b[1] is None and b[0].numel() >= need]
            if free:
                return free[0]
            idle = [k for k, b in enumerate(self._bufs) if b[1] is None]
            if len(self._bufs) < self.MAX_BUFS or idle:
                if idle:   # too small: replace it
                    self._release(idle[0])
                    k = idle[0]
                else:
                    self._bufs.append(None)
                    k = len(self._bufs) - 1
                t = torch.empty(int(need * 1.05) + 4096, dtype=torch.uint8).share_memory_()
                if torch.cuda.is_available():   # page-locked: device-to-host copies at full speed
                    try:
                        torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)
                    except Exception:
                        pass
                self._bufs[k] = [t, None]
                return k
            self._reap(block=True)

    def _release(self, k):
        if torch.cuda.is_available():
            try:
                torch.cuda.cudart().cudaHostUnregister(self._bufs[k][0].data_ptr())
            except Exception:
                pass

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise TypeError("AsyncShmCheckpointIO takes no storage_options")
        t_snap = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        if self._proc is None:
            _clean_tmp(os.path.dirname(os.path.abspath(str(path))))
            ctx = mp.get_context("spawn")
            self._jobs, self._acks = ctx.Queue(), ctx.Queue()
            self._proc = ctx.Process(target=_writer, args=(self._jobs, self._acks), daemon=True)
            self._proc.start()
            if not getattr(self, "_atexit", False):
                atexit.register(self._teardown_at_exit)
                self._atexit = True
        twin = _twins(checkpoint)
        one = lambda t: twin.get(id(t), t)   # the tensor whose bytes stand for t: a twin's raw tensor, else t itself
        distinct = {}
        apply_to_collection(checkpoint, torch.Tensor, lambda t: distinct.setdefault(id(one(t)), one(t)))
        align = lambda n: (n + 255) // 256 * 256
        k = self._buffer(sum(align(t.numel() * t.element_size()) for t in distinct.values()))
        buf, off, refs = self._bufs[k][0], [0], {}

        def put(t):
            t = one(t)
            if id(t) not in refs:   # one copy per distinct tensor
                n = t.numel() * t.element_size()
                buf[off[0]:off[0] + n].view(t.dtype).view(t.shape).copy_(t.detach(), non_blocking=t.is_cuda)
                refs[id(t)] = (_Ref(off[0], t.dtype, t.shape), t)   # t kept: an id is unique only while its object lives
                off[0] += align(n)
            return refs[id(t)][0]
        snap = apply_to_collection(checkpoint, torch.Tensor, put)
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # the copies are complete before training changes the tensors
        self._bufs[k][1] = str(path)
        self._jobs.put((k, buf, snap, str(path), t_snap))

    def _teardown_at_exit(self):
        try:
            self.teardown()
        except Exception as e:   # a lost final checkpoint must not end the job with exit 0
            print(f"AsyncShmCheckpointIO at exit: {e!r}", file=sys.stderr, flush=True)
            if self._proc is not None and self._proc.is_alive():   # the writer ignores SIGTERM (S3): kill it
                self._proc.kill()
            os._exit(97)

    def teardown(self):
        if self._proc is not None:
            while any(b[1] is not None for b in self._bufs):
                self._reap(block=True)
            self._jobs.put(None)
            self._proc.join()
            self._proc = None
        for k in range(len(self._bufs)):
            self._release(k)
        self._bufs = []
