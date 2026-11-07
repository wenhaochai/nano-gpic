"""MFU lever E2 (GPIC_MFU_REBALANCE=2 with GPIC_MFU_PACK=1 and GPIC_MFU_GPU_RESIZE=1, docs/mfu.md): lever E's deal of the
samples over the ranks, done on the host one step ahead, through shared memory (every rank of a ladder run is on one node).

Lever E planned on the host inside the training step (a blocking gloo all_gather) and moved the samples with two NCCL
all_to_all calls at the start of the step, which line up all ranks there; its net effect was +2 to +3 ms per step while
the imbalance it removes is about 7 ms at d1024 (w1315). Here a thread of each rank takes the next batch from the data
loader, writes its samples (the source crop and resize tables of lever C, the caption embedding rows of lever E's packed
collate) into this rank's shared-memory slot, and joins one gloo all_gather of the batch's caption lengths and the
samples' byte ranges; that all_gather is also the barrier that says every rank's slot is written. The plan is
src/utils/rebalance.plan (the same deal: at each batch position the 8 samples there go to the 8 ranks, longest first to
the least loaded), so a sample keeps its batch position, and with it its t and noise. The thread then copies the samples
this rank computes out of the 8 slots and builds the batch the collate would have built for them (src/data/dataset/
pil_exact_gpu.pack format, packed caption rows, lengths, caption types), pinned; the training step sees an ordinary batch
and builds y_pack without any exchange. Two slots per rank: a rank writes slot s % 2 for step s only after the all_gather
of step s - 1, by which time every rank has copied its samples of step s - 2 out of that slot.
No device work and no GPU synchronization: the step's GPU timeline has no exchange in it.
"""
import os
import queue
import threading

import numpy as np
import torch
import torch.distributed as dist

from src.utils.rebalance import plan

SLOT_BYTES = int(os.environ.get("GPIC_MFU_REBALANCE_SLOT_MB", "256")) << 20


def _align(n, a=256):
    return (n + a - 1) // a * a


def split_samples(batch):
    """The per-sample pieces of a packed batch: (src bytes uint8, xmin [2, OUT] int32, kk [2, OUT, K] int32,
    hw [3, 2] int64, axis [2] int64), caption embedding rows [n, D] bf16, caption, caption type."""
    job, caps, meta = batch
    B = job["B"]
    src, off = job["src"], job["src_off"].tolist()
    ends = off[1:] + [src.numel()]
    lens = list(meta["text_lens"])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int).tolist()
    hw, axis = job["hw"], job["axis"]
    out = []
    for b in range(B):
        out.append({"src": src[off[b]:ends[b]], "xmin": job["xmin"][b], "kk": job["kk"][b], "hw": hw[b], "axis": axis[b],
                    "emb": meta["text_emb"][starts[b]:starts[b] + lens[b]], "cap": caps[b],
                    "type": meta["caption_type"][b], "len": lens[b]})
    return out


def join_samples(samples, pin):
    """The inverse of split_samples for a new list of samples: the dict pil_exact_gpu.pack builds, and the packed
    collate's caption rows, lengths and types (pinned when pin)."""
    from src.data.dataset.pil_exact_gpu import OUT, _geometry
    B = len(samples)
    K = max(s["kk"].shape[-1] for s in samples)
    sizes = [s["src"].numel() for s in samples]
    off = np.zeros(B, dtype=np.int64)
    off[1:] = np.cumsum(sizes)[:-1]
    flat = torch.empty(sum(sizes), dtype=torch.uint8, pin_memory=pin)
    xmin = torch.empty(B, 2, OUT, dtype=torch.int32, pin_memory=pin)
    kk = torch.zeros(B, 2, OUT, K, dtype=torch.int32, pin_memory=pin)
    hw = np.zeros((B, 3, 2), dtype=np.int64)
    axis = np.zeros((B, 2), dtype=np.int64)
    for b, s in enumerate(samples):
        flat[off[b]:off[b] + sizes[b]].copy_(s["src"])
        xmin[b].copy_(s["xmin"])
        kk[b, :, :, :s["kk"].shape[-1]].copy_(s["kk"])
        hw[b], axis[b] = s["hw"].numpy() if torch.is_tensor(s["hw"]) else s["hw"], \
            s["axis"].numpy() if torch.is_tensor(s["axis"]) else s["axis"]
    g1, mid_bytes = _geometry(hw, axis, 0)
    g2, _ = _geometry(hw, axis, 1)
    mid_off = np.zeros(B, dtype=np.int64)
    mid_off[1:] = np.cumsum(mid_bytes)[:-1]
    job = {"src": flat, "src_off": torch.from_numpy(off), "xmin": xmin, "kk": kk,
           "geo": torch.from_numpy(np.stack([g1, g2])), "mid_off": torch.from_numpy(mid_off),
           "out_off": torch.arange(B, dtype=torch.int64) * (OUT * OUT * 3), "K": int(K), "B": B,
           "mid_bytes": int(mid_bytes.sum()), "n_i_max": [int(g1[0].max()), int(g2[0].max())],
           "hw": torch.from_numpy(hw), "axis": torch.from_numpy(axis)}
    lens = [s["len"] for s in samples]
    D = samples[0]["emb"].shape[-1]
    emb = torch.empty(sum(lens), D, dtype=samples[0]["emb"].dtype, pin_memory=pin)
    o = 0
    for s in samples:
        emb[o:o + s["len"]].copy_(s["emb"])
        o += s["len"]
    meta = {"caption_type": [s["type"] for s in samples], "text_emb": emb, "text_lens": lens}
    return job, [s["cap"] for s in samples], meta


class _Slots:
    """This rank's two shared-memory slots (written by it) and every other rank's (read by it)."""

    def __init__(self, rank, world, tag, group):
        from multiprocessing import shared_memory, resource_tracker
        self.own, self.all = [], {}
        for k in range(2):
            name = f"gpic_rb_{tag}_{rank}_{k}"
            try:   # a run that died before its exit hook left it behind
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            shm = shared_memory.SharedMemory(name=name, create=True, size=SLOT_BYTES)
            self.own.append(shm)
            self.all[(rank, k)] = shm
        dist.barrier(group=group)
        for r in range(world):
            for k in range(2):
                if r != rank:
                    shm = shared_memory.SharedMemory(name=f"gpic_rb_{tag}_{r}_{k}")
                    resource_tracker.unregister(shm._name, "shared_memory")   # its owner unlinks it, not this process
                    self.all[(r, k)] = shm
        self.view = {key: torch.frombuffer(shm.buf, dtype=torch.uint8) for key, shm in self.all.items()}

    def close(self):
        """Unlink this rank's slots (the name goes; the memory stays mapped until every process lets go, so a thread
        still copying is safe), then unmap what nothing points into any more."""
        for shm in self.own:
            try:
                shm.unlink()
            except Exception:
                pass
        self.view = {}
        for shm in self.all.values():
            try:
                shm.close()
            except Exception:   # a tensor still views it: the process exit unmaps it
                pass


_FIELDS = ("src", "xmin", "kk", "hw", "axis", "emb")


def _write(view, samples):
    """Samples into a slot, back to back; returns per sample {field: (offset, shape, dtype)} plus its scalars."""
    o, out = 0, []
    for s in samples:
        d = {"cap": s["cap"], "type": s["type"], "len": s["len"]}
        for f in _FIELDS:
            t = s[f] if torch.is_tensor(s[f]) else torch.from_numpy(np.ascontiguousarray(s[f]))
            t = t.contiguous()
            n = t.numel() * t.element_size()
            if o + n > view.numel():
                raise RuntimeError(f"rebalance slot of {view.numel() >> 20} MiB too small (GPIC_MFU_REBALANCE_SLOT_MB)")
            view[o:o + n].view(t.dtype).view(t.shape).copy_(t) if n else None
            d[f] = (o, tuple(t.shape), str(t.dtype).replace("torch.", ""))
            o = _align(o + n)
        out.append(d)
    return out


def _read(view, d):
    s = {"cap": d["cap"], "type": d["type"], "len": d["len"]}
    for f in _FIELDS:
        o, shape, dt = d[f]
        dtype = getattr(torch, dt)
        n = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
        s[f] = view[o:o + n].view(dtype).view(shape) if n else torch.empty(shape, dtype=dtype)
    return s


def write_batch(view, batch):
    """The whole batch into a slot in six bulk copies (the job's flat crops, the two passes' tables, the geometry, the
    caption rows); returns the descriptor the other ranks need: per-sample byte ranges and the regions' layout."""
    job, caps, meta = batch
    regions, o = {}, 0
    for name, t in (("src", job["src"]), ("xmin", job["xmin"]), ("kk", job["kk"]), ("hw", job["hw"]),
                    ("axis", job["axis"]), ("emb", meta["text_emb"])):
        t = t.contiguous()
        n = t.numel() * t.element_size()
        if o + n > view.numel():
            raise RuntimeError(f"rebalance slot of {view.numel() >> 20} MiB too small (GPIC_MFU_REBALANCE_SLOT_MB)")
        if n:
            view[o:o + n].view(t.dtype).view(t.shape).copy_(t)
        regions[name] = (o, tuple(t.shape), str(t.dtype).replace("torch.", ""))
        o = _align(o + n)
    lens = [int(x) for x in meta["text_lens"]]
    off = job["src_off"].tolist()
    sizes = [b - a for a, b in zip(off, off[1:] + [job["src"].numel()])]
    return {"regions": regions, "src_off": off, "src_size": sizes, "lens": lens,
            "emb_start": np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int).tolist(),
            "caps": list(caps), "types": list(meta["caption_type"])}


def _region(view, reg):
    o, shape, dt = reg
    dtype = getattr(torch, dt)
    n = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    return view[o:o + n].view(dtype).view(shape) if n else torch.empty(shape, dtype=dtype)


def assemble_batch(views, infos, mine, pin):
    """This rank's batch from the slots: mine = [(source rank, position)] in position order. The same batch
    join_samples builds from the same samples, with one cat or indexed copy per field and source rank."""
    from src.data.dataset.pil_exact_gpu import OUT, _geometry
    B = len(mine)
    R = {r: {k: _region(views[r], v) for k, v in infos[r]["regions"].items()} for r in {r for r, _ in mine}}
    sizes = [infos[r]["src_size"][i] for r, i in mine]
    off = np.zeros(B, dtype=np.int64)
    off[1:] = np.cumsum(sizes)[:-1]
    flat = torch.empty(sum(sizes), dtype=torch.uint8, pin_memory=pin)
    torch.cat([R[r]["src"][infos[r]["src_off"][i]:infos[r]["src_off"][i] + infos[r]["src_size"][i]] for r, i in mine],
              out=flat)
    K = max(R[r]["kk"].shape[-1] for r in R)
    xmin = torch.empty(B, 2, OUT, dtype=torch.int32, pin_memory=pin)
    kk = torch.zeros(B, 2, OUT, K, dtype=torch.int32, pin_memory=pin)
    hw = torch.empty(B, 3, 2, dtype=torch.int64)
    axis = torch.empty(B, 2, dtype=torch.int64)
    for r in R:   # one indexed copy per field and source rank
        pos = [b for b, (rr, _) in enumerate(mine) if rr == r]
        idx = torch.tensor([i for rr, i in mine if rr == r])
        p = torch.tensor(pos)
        xmin[p] = R[r]["xmin"][idx]
        kk[p, :, :, :R[r]["kk"].shape[-1]] = R[r]["kk"][idx]
        hw[p] = R[r]["hw"][idx]
        axis[p] = R[r]["axis"][idx]
    hw_np, axis_np = hw.numpy(), axis.numpy()
    g1, mid_bytes = _geometry(hw_np, axis_np, 0)
    g2, _ = _geometry(hw_np, axis_np, 1)
    mid_off = np.zeros(B, dtype=np.int64)
    mid_off[1:] = np.cumsum(mid_bytes)[:-1]
    job = {"src": flat, "src_off": torch.from_numpy(off), "xmin": xmin, "kk": kk,
           "geo": torch.from_numpy(np.stack([g1, g2])), "mid_off": torch.from_numpy(mid_off),
           "out_off": torch.arange(B, dtype=torch.int64) * (OUT * OUT * 3), "K": int(K), "B": B,
           "mid_bytes": int(mid_bytes.sum()), "n_i_max": [int(g1[0].max()), int(g2[0].max())], "hw": hw, "axis": axis}
    lens = [infos[r]["lens"][i] for r, i in mine]
    D = R[mine[0][0]]["emb"].shape[-1]
    emb = torch.empty(sum(lens), D, dtype=R[mine[0][0]]["emb"].dtype, pin_memory=pin)
    torch.cat([R[r]["emb"][infos[r]["emb_start"][i]:infos[r]["emb_start"][i] + infos[r]["lens"][i]] for r, i in mine],
              out=emb)
    meta = {"caption_type": [infos[r]["types"][i] for r, i in mine], "text_emb": emb, "text_lens": lens}
    if pin:   # every tensor pinned, as the loader's pin thread does: a pageable copy to the GPU blocks the host (perf C7)
        job = {k: (v.pin_memory() if torch.is_tensor(v) and not v.is_pinned() else v) for k, v in job.items()}
    return job, [infos[r]["caps"][i] for r, i in mine], meta


class RebalancingLoader:
    """Wraps the training loader: iterating it yields rebalanced batches (see the module doc)."""

    def __init__(self, loader, depth=2):
        self.loader, self.depth = loader, depth

    def __iter__(self):
        return _Iter(self.loader, self.depth)

    def __getattr__(self, k):
        return getattr(self.loader, k)


class _Iter:
    """Starts at its first __next__, not at __iter__: Lightning calls iter() once just to check that the loader is
    iterable and drops the result (w1420: a thread started there took every other batch from the shared, persistent
    worker iterator, so the steps got other samples than the reference arm, and the workers loaded twice as much)."""
    _active = None

    def __init__(self, loader, depth):
        self.loader, self.depth, self.started = loader, depth, False

    def _start(self):
        assert dist.is_initialized()
        assert _Iter._active is None, "one rebalancing iterator per process: the loader's worker iterator is shared"
        _Iter._active = self
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.group = dist.new_group(backend="gloo")
        tag = os.environ.get("WQ_JOB_ID") or os.environ.get("SLURM_JOB_ID") or str(os.getppid())
        tag = f"{tag}_{os.environ.get('MASTER_PORT', '0')}"
        self.slots = _Slots(self.rank, self.world, tag, self.group)
        import atexit
        atexit.register(self.slots.close)
        self.pin = torch.cuda.is_available()
        self.q = queue.Queue(maxsize=self.depth)
        self.balance = []   # per step: the busiest rank's caption tokens over the mean
        self.thread = threading.Thread(target=self._run, args=(iter(self.loader),), daemon=True)
        self.thread.start()
        self.started = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.started:
            self._start()
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def _run(self, it):
        s = 0
        try:
            while True:
                batch = next(it)
                k = s % 2
                desc = write_batch(self.slots.view[(self.rank, k)], batch)
                infos = [None] * self.world
                dist.all_gather_object(infos, desc, group=self.group)   # also: every rank's slot k is written
                all_lens = [inf["lens"] for inf in infos]
                B = len(all_lens[self.rank])
                assert all(len(l) == B for l in all_lens), [len(l) for l in all_lens]
                _, _, _, recv = plan(all_lens, self.rank, self.world, B)
                mine = sorted(recv, key=lambda ri: ri[1])   # (source rank, position) in position order
                out = assemble_batch({r: self.slots.view[(r, k)] for r in range(self.world)}, infos, mine, self.pin)
                loads = [sum(l) for l in all_lens]
                self.balance.append((max(loads) / (sum(loads) / self.world), None))
                self.q.put(out)
                s += 1
        except BaseException as e:   # surfaced in the training loop by __next__
            self.q.put(e)
