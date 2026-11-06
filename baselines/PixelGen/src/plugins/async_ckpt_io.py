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
Use: trainer.plugins += [src.plugins.async_ckpt_io.AsyncShmCheckpointIO] (GPIC_MFU_ASYNC_CKPT=1 in jobs/mfu/arm.sh).
"""
import atexit
import os
import queue
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


def _writer(jobs, acks):
    """The writer process: rebuild the tensors from the shared buffer, torch.save, rename, set the mtime."""
    while True:
        job = jobs.get()
        if job is None:
            return
        k, buf, snap, path, t_snap = job
        try:
            delay = float(os.environ.get("GPIC_ASYNC_CKPT_TEST_DELAY", "0"))   # tests only: a slow file system
            if delay:
                time.sleep(delay)
            def rebuild(r):
                n = int(torch.Size(r.shape).numel()) * torch.empty((), dtype=r.dtype).element_size()
                return buf[r.off:r.off + n].view(r.dtype).view(r.shape).clone()
            ckpt = apply_to_collection(snap, _Ref, rebuild)
            tmp = f"{path}.tmp{os.getpid()}"
            torch.save(ckpt, tmp)
            os.replace(tmp, path)
            os.utime(path, (t_snap, t_snap))
            del ckpt
            acks.put((k, path, None))
        except Exception as e:   # raised in the training process at its next save or at teardown
            acks.put((k, path, repr(e)))
        del buf


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
            ctx = mp.get_context("spawn")
            self._jobs, self._acks = ctx.Queue(), ctx.Queue()
            self._proc = ctx.Process(target=_writer, args=(self._jobs, self._acks), daemon=True)
            self._proc.start()
            if not getattr(self, "_atexit", False):
                atexit.register(self._teardown_at_exit)
                self._atexit = True
        tensors = []
        apply_to_collection(checkpoint, torch.Tensor, lambda t: tensors.append(t))
        align = lambda n: (n + 255) // 256 * 256
        k = self._buffer(sum(align(t.numel() * t.element_size()) for t in tensors))
        buf, off = self._bufs[k][0], [0]

        def put(t):
            n = t.numel() * t.element_size()
            buf[off[0]:off[0] + n].view(t.dtype).view(t.shape).copy_(t.detach(), non_blocking=t.is_cuda)
            r = _Ref(off[0], t.dtype, t.shape)
            off[0] += align(n)
            return r
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
