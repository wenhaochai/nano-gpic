"""Repetition arms (owner 2026-11-07): CPU tests of the shard copies of EpochShuffledShards (k > 1) and of the unchanged
path (k = 1). Run on a vis node:
  cd <worktree> && PYTHONPATH=baselines/PixelGen CUDA_VISIBLE_DEVICES= python tools/test_repeat.py
1a. every worker of every rank gets the same number of entries per pass (8 shards: 1, 80 shards: 5) and the same number
    of samples, every shard has k copies per pass, and the global step's 8 batches never hold one shard twice
1a'. real data: an entry read through _rotated_tar_samples gives L samples, byte-identical to the same samples of the
    plain tar, starting at its start sample and wrapping; the full training pipeline (text cache, crop, grouping) gives
    batches from a k > 1 entry; GPICWebDataset picks the copies only for the training set with fewer shards than slots
1b. the copies of a shard in one slot start evenly spaced (1 / m of the shard apart); the start moves every pass
1c. resume with k > 1: a stopped launch logs "<url>#<copy>:<k>" entries; the resumed launch reads none of them and every
    other entry of the pass once, spread over all workers; a pass with all entries started resumes in the next pass;
    a torn last line is dropped; a log from another seed or another number of workers stops the resume
1d. bit-identity for n >= world x workers (GPIC Full 8000 shards, Lite 801, the boundary 256) and without workers:
    against the code of 5f3c700 (git show), the same permutations, the same entries for every rank and worker, the same
    log lines, the same resume points and the same pipeline stages; on real shards the same samples in the same order
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from itertools import islice

import numpy as np
import torch
import torch.utils.data as tud

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "baselines", "PixelGen"))
import webdataset as wds  # noqa: E402
from src.data.dataset import gpic_dataset as new  # noqa: E402
from src.data.dataset.gpic_dataset import EpochShuffledShards, GPICWebDataset, _rotated_tar_samples, tar_sample_offsets  # noqa: E402

WORLD, NW, SEED = 8, 32, 20261106
TAR_DIR = "/scratch/gpfs/GROUP/USER/data/gpic/train/"
EMB_DIR = "/scratch/gpfs/GROUP/USER/data/gpic/train_emb_qwen3_1.7b"
REAL = [f"{TAR_DIR}gpic_train_{i:05d}.tar" for i in range(80)]
BASE = "5f3c700"


class _WI:
    def __init__(self, i, n):
        self.id, self.num_workers = i, n


def as_rank(r, f, world=WORLD):
    old = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE")}
    os.environ.update(RANK=str(r), WORLD_SIZE=str(world))
    try:
        return f()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def as_worker(w, f, nw=NW):
    old = tud.get_worker_info
    tud.get_worker_info = lambda: _WI(w, nw)
    try:
        return f()
    finally:
        tud.get_worker_info = old


def load_old():
    """gpic_dataset.py of the base commit, as a module of the same package (its relative import works)."""
    src = subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:baselines/PixelGen/src/data/dataset/gpic_dataset.py"],
                         capture_output=True, text=True, check=True).stdout
    d = tempfile.mkdtemp()
    p = os.path.join(d, "gpic_dataset_base.py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("src.data.dataset.gpic_dataset_base", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def worker_entries(copies, r, w):
    """What worker w of rank r reads in one pass, on the training path: k > 1 splits inside the class, k = 1 after it."""
    c = copies[w]
    if c.k > 1:
        return as_rank(r, lambda: as_worker(w, lambda: list(iter(c))))
    return as_rank(r, lambda: as_worker(w, lambda: list(islice(iter(c), w, None, NW))))


def main():
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {detail}  {'ok' if good else 'FAIL'}", flush=True)

    # 1a. coverage: every worker the same number of entries and samples; k copies of each shard; no shard twice in a step
    for n in (8, 80):
        urls = REAL[:n]
        per_worker, names, step_ok, len_ok = set(), [], True, True
        L = None
        for r in range(WORLD):
            c0 = as_rank(r, lambda: EpochShuffledShards(urls, SEED, workers=NW))
            if r == 0:
                c0.load_index()
                index, L = c0.index, c0.length
            for w in range(NW):   # a fresh copy per worker, as the fork gives each worker the rank's state
                c = as_rank(r, lambda: EpochShuffledShards(urls, SEED, workers=NW))
                c.index, c.length = index, L
                ents = worker_entries([c] * NW, r, w)
                per_worker.add(len(ents))
                names += [e["entry"] for e in ents]
                for e in ents:   # the samples each entry gives: L, in one or two byte ranges of whole samples
                    st, end = c.index[e["url"]]
                    nb = sum(b - a for a, b in e["parts"])
                    a = min(int(e["start"] * len(st)), len(st) - 1)
                    idx = [(a + q) % len(st) for q in range(L)]
                    want = sum(int((st[i + 1] if i + 1 < len(st) else end) - st[i]) for i in idx)
                    len_ok &= nb == want
        k = c0.k
        sched = c0._schedule(0)
        # worker w of every rank reads entries perm[r::8][w::32]: the 8 batches of global step s come from one slot entry
        # per rank, entries 8 (32 j + w) .. 8 (32 j + w) + 7 of the schedule
        for j in range(0, len(sched), WORLD):
            blk = [u for _, u, _ in sched[j:j + WORLD]]
            step_ok &= len(set(blk)) == len(blk)
        counts = {u: sum(1 for x in names if x.rsplit("#", 1)[0] == u) for u in urls}
        good = (per_worker == {k * n // (WORLD * NW)} and len(names) == len(set(names)) == k * n
                and set(counts.values()) == {k} and step_ok and len_ok)
        check(f"1a. {n} shards", good, f"k = {k} copies per shard, every one of 8 x 32 workers gets {sorted(per_worker)} "
              f"entries of {L} samples each, {len(names)} entries each once per pass, no shard twice in a global step {step_ok}, "
              f"byte ranges cover exactly L samples {len_ok}")

    # 1a'. real bytes: a k > 1 entry gives L samples, the same bytes as the plain tar, from its start sample, wrapping
    c = as_rank(0, lambda: EpochShuffledShards(REAL[:8], SEED, workers=NW))
    c.load_index()
    u = REAL[3]
    st, _ = c.index[u]
    plain = list(wds.DataPipeline(wds.SimpleShardList([u]), wds.tarfile_to_samples()))
    by_key = {s["__key__"]: s for s in plain}
    start = 0.99   # wraps past the end
    d = dict(url=u, shard_pass=0, entry=f"{u}#0", start=start, parts=c.parts(u, start))
    got = list(_rotated_tar_samples([d]))
    a = min(int(start * len(st)), len(st) - 1)
    want_keys = [plain[(a + q) % len(plain)]["__key__"] for q in range(c.length)]
    same = [s["__key__"] for s in got] == want_keys and all(
        all(s[x] == by_key[s["__key__"]][x] for x in s if not x.startswith("__")) for s in got)
    check("1a'. rotated read", same and len(plain) == len(st),
          f"{len(got)} samples (L = {c.length} of {len(plain)}) from sample {a}, wrapping, byte-identical to the plain tar")

    # the full training pipeline on a k > 1 entry: rank 0 of 8, worker 5 of 32, text cache, crop, grouping
    os.environ.update(GPIC_SHARD_SEED=str(SEED), GPIC_REGION_CROP="1")
    os.environ.pop("GPIC_SHARD_LOG_DIR", None)
    try:
        ds = GPICWebDataset(tar_dir=TAR_DIR, tar_pattern="gpic_train_{00000..00007}.tar", group_text_len=True,
                            group_window=1024, group_batch=32, pad_multiple=8, emb_dir=EMB_DIR)
        ds._num_workers = NW
        pipe = as_rank(0, lambda: ds.create_dataset())
        stages = pipe.pipeline
        samples = as_rank(0, lambda: as_worker(5, lambda: list(islice(iter(pipe), 64))))
        batch = ds._collate_cached(samples[:32])
        good = ds._repeat_arm and stages[1] is _rotated_tar_samples and len(samples) == 64 and batch[0].shape == (32, 3, 256, 256) and batch[2]["text_emb"].dtype == torch.bfloat16
        check("1a'. training pipeline", good, f"8-shard pattern: copies on ({ds._repeat_arm}); 64 samples through the "
              f"text cache, crop and grouping; a batch {tuple(batch[0].shape)}, text {tuple(batch[2]['text_emb'].shape)}")
        va = GPICWebDataset(tar_dir=TAR_DIR, tar_pattern="gpic_train_{00000..00007}.tar", is_val=True)
        va._num_workers = NW
        va_st = as_rank(0, lambda: va._shard_stages(TAR_DIR + "gpic_train_{00000..00007}.tar"))
        check("1a'. validation set", not va._repeat_arm and va_st[0].k == 1, "the validation set never takes copies")
    finally:
        os.environ.pop("GPIC_SHARD_SEED", None)

    # 1b. starts: the m copies of a shard in one slot are 1 / m apart; slot g + 1 starts L samples after slot g
    prev_m = {}
    for n in (8, 80):
        c = as_rank(0, lambda: EpochShuffledShards(REAL[:n], SEED, workers=NW))
        c.load_index()
        S = c.k * n // (WORLD * NW)
        spaced, carried = True, True
        prev = {}
        for p in (0, 1, 2):
            s = c._schedule(p)
            for t, j in enumerate(range(0, len(s), WORLD * NW)):
                slot = s[j:j + WORLD * NW]
                for u in {x[1] for x in slot}:
                    fr = sorted(x[2] for x in slot if x[1] == u)
                    gaps = np.diff(fr + [fr[0] + 1])
                    spaced &= np.allclose(gaps, 1 / len(fr))
                    g = p * S + t
                    step = c.length / len(c.index[u][0])
                    if u in prev:   # the slot's first start set moved by (g - g') L / N against the shard's last slot
                        g0, f0 = prev[u]
                        d = (fr[0] - f0 - (g - g0) * step) % (1 / len(fr))
                        carried &= min(d, 1 / len(fr) - d) < 1e-9 or len(fr) != prev_m[u]
                    prev[u] = (g, fr[0])
                    prev_m[u] = len(fr)
        m = (WORLD * NW) // n
        check(f"1b. {n} shards", spaced and carried, f"the copies of a shard in a slot start 1/m of the shard apart "
              f"(m = {m}{' or ' + str(m + 1) if (WORLD * NW) % n else ''}); each slot's starts move L / N past the last one, "
              f"across passes too")

    # 1c. resume with k > 1 on the 80-shard subset (5 entries per worker): a stop in slot 2, some workers in slot 3
    urls = REAL[:80]
    with tempfile.TemporaryDirectory() as d:
        ds_log = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x")
        started, idx = {}, {}
        for r in range(WORLD):
            c0 = as_rank(r, lambda: EpochShuffledShards(urls, SEED, d, workers=NW))
            c0.load_index() if r == 0 else None
            idx = idx or (c0.index, c0.length)
            copies = [as_rank(r, lambda: EpochShuffledShards(urls, SEED, d, workers=NW)) for _ in range(NW)]
            for w in range(NW):
                c = copies[w]
                c.index, c.length = idx
                k_read = 3 + (w % 7 == 0)   # 3 entries started (slot 0, 1, 2), every 7th worker already in slot 3

                def launch1():
                    ds_log._shard_log = c.log_path
                    g = ds_log._log_started(iter(c))
                    return [next(g)["entry"] for _ in range(k_read)]
                started[(r, w)] = as_rank(r, lambda: as_worker(w, launch1))
        skip_ok, cover_ok, spread_ok, next_ok, print_ok = True, True, True, True, True
        for r in range(WORLD):
            rows = [x.split() for x in open(os.path.join(d, f"shards_started_rank{r}.txt"))]
            logged = {u for _, _, u in rows}
            again, per_w, nxt = [], [], []
            for w in range(NW):
                c2 = as_rank(r, lambda: EpochShuffledShards(urls, SEED, d, workers=NW))
                c2.index, c2.length = idx
                got = as_rank(r, lambda: as_worker(w, lambda: [x["entry"] for x in iter(c2)]))
                again += got
                per_w.append(len(got))
                nxt += as_rank(r, lambda: as_worker(w, lambda: [x["entry"] for x in iter(c2)]))
            mine = [e[0] for e in c2._schedule(0)[r::WORLD]]
            skip_ok &= not set(again) & logged and logged == {e for w in range(NW) for e in started[(r, w)]}
            cover_ok &= sorted(list(logged) + again) == sorted(mine) and len(set(again)) == len(again)
            spread_ok &= max(per_w) - min(per_w) <= 1 and min(per_w) >= 1
            next_ok &= sorted(nxt) == sorted(e[0] for e in c2._schedule(1)[r::WORLD])
            print_ok &= all(x.count("#") == 1 and x.endswith(":16") for x in logged)
        check("1c. resume", skip_ok and cover_ok and spread_ok and next_ok and print_ok,
              "the log names <url>#<copy>:<k>; the resumed launch reads no logged entry, every other entry of the pass once, "
              "over all 32 workers (counts within 1); the next pass is whole")

    with tempfile.TemporaryDirectory() as d:   # every entry of pass 0 started: resume in pass 1; a torn line; a foreign log
        r = 4
        c = as_rank(r, lambda: EpochShuffledShards(REAL[:8], SEED, workers=NW))
        mine0, mine1 = c._names(0), c._names(1)
        f = os.path.join(d, f"shards_started_rank{r}.txt")
        with open(f, "w") as fh:
            fh.writelines(f"{100 + i}.000 0 {u}\n" for i, u in enumerate(mine0))
        c = as_rank(r, lambda: EpochShuffledShards(REAL[:8], SEED, d, workers=NW))
        whole = (c.passes, c.skip) == (1, set())
        with open(f, "a") as fh:
            fh.writelines(f"{200 + i}.000 1 {u}\n" for i, u in enumerate(mine1[:3]))
            fh.write(f"300.000 1 {mine1[3][:-1]}")   # torn inside "<url>#<copy>:<k>": still 3 fields, no newline
        c = as_rank(r, lambda: EpochShuffledShards(REAL[:8], SEED, d, workers=NW))
        torn = (c.passes, c.skip) == (1, set(mine1[:3]))
        try:
            as_rank(r, lambda: EpochShuffledShards(REAL[:8], SEED, d, workers=16))   # 16 workers: k 16, other names
            foreign_w = False
        except AssertionError:
            foreign_w = True
        try:
            as_rank(r, lambda: EpochShuffledShards(REAL[:8], SEED + 1, d, workers=NW))
            foreign_s = False
        except AssertionError:
            foreign_s = True
        check("1c. pass ends, torn line, foreign log", whole and torn and foreign_w and foreign_s,
              "all of pass 0 logged: pass 1; a torn <url>#<copy>:<k> line dropped; another number of workers or seed stops the resume")

    # 1d. bit-identity with the base commit when n >= world x workers, and without workers
    old = load_old()
    for n in (8000, 801, 256):
        urls = [f"/d/gpic_train_{i:05d}.tar" for i in range(n)]
        same = True
        for r in range(WORLD):
            co = as_rank(r, lambda: old.EpochShuffledShards(urls, SEED))
            cn = as_rank(r, lambda: EpochShuffledShards(urls, SEED, workers=NW))
            same &= cn.k == 1 and co._perm(0) == cn._perm(0) and co._perm(5) == cn._perm(5)
            for w in range(0, NW, 5):
                co = as_rank(r, lambda: old.EpochShuffledShards(urls, SEED))
                cn = as_rank(r, lambda: EpochShuffledShards(urls, SEED, workers=NW))
                for _ in range(2):   # two passes
                    a = as_rank(r, lambda: as_worker(w, lambda: list(islice(iter(co), w, None, NW))))
                    b = worker_entries([cn] * NW, r, w)
                    same &= a == b
        # logs and resume: the same lines (time aside) and the same resume points
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            dso = old.GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x")
            dsn = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x")
            for r in range(WORLD):
                for w in range(3):
                    co = as_rank(r, lambda: old.EpochShuffledShards(urls, SEED, d1))
                    cn = as_rank(r, lambda: EpochShuffledShards(urls, SEED, d2, workers=NW))
                    dso._shard_log, dsn._shard_log = co.log_path, cn.log_path
                    take = 1 + (r + w) % 3
                    as_rank(r, lambda: list(islice(dso._log_started(islice(iter(co), w, None, NW)), take)))
                    as_rank(r, lambda: list(islice(dsn._log_started(worker_entries([cn] * NW, r, w)), take)))
            for r in range(WORLD):
                lo = [x.split()[1:] for x in open(os.path.join(d1, f"shards_started_rank{r}.txt"))]
                ln = [x.split()[1:] for x in open(os.path.join(d2, f"shards_started_rank{r}.txt"))]
                same &= lo == ln
                co = as_rank(r, lambda: old.EpochShuffledShards(urls, SEED, d1))
                cn = as_rank(r, lambda: EpochShuffledShards(urls, SEED, d2, workers=NW))
                same &= (co.passes, co.skip) == (cn.passes, cn.skip)
                a = as_rank(r, lambda: as_worker(7, lambda: list(islice(iter(co), 7, None, NW))))
                b = worker_entries([cn] * NW, r, 7)
                same &= a == b
        check(f"1d. {n} shards", same, "k = 1; permutations, entries of every rank and worker over two passes, log lines, "
              "resume points and the resumed entries equal the base commit's")
    urls = [f"/d/gpic_train_{i:05d}.tar" for i in range(8)]   # no workers given (validation set, tests): as before
    co = as_rank(2, lambda: old.EpochShuffledShards(urls, SEED))
    cn = as_rank(2, lambda: EpochShuffledShards(urls, SEED))
    check("1d. no workers", cn.k == 1 and as_rank(2, lambda: list(iter(co))) == as_rank(2, lambda: list(iter(cn))),
          "8 shards without a worker count: the base commit's entries")

    # the same pipeline stages and, on 256 real shards (the boundary, k = 1), the same samples in the same order
    os.environ.update(GPIC_SHARD_SEED=str(SEED))
    try:
        pat = "gpic_train_{00000..00255}.tar"
        dso = old.GPICWebDataset(tar_dir=TAR_DIR, tar_pattern=pat)
        dsn = GPICWebDataset(tar_dir=TAR_DIR, tar_pattern=pat)
        dsn._num_workers = NW
        po = as_rank(3, lambda: dso.create_dataset())
        pn = as_rank(3, lambda: dsn.create_dataset())
        nm = lambda x: getattr(x, "__name__", type(x).__name__)
        desc = lambda p: [(nm(s), nm(getattr(s, "f", s)), [nm(a) if callable(a) else a for a in getattr(s, "args", ())],
                           {k: nm(v) if callable(v) else v for k, v in getattr(s, "kw", {}).items()}) for s in p.pipeline]
        stages_same = desc(po) == desc(pn) and not dsn._repeat_arm
        so = as_rank(3, lambda: dso._shard_stages(TAR_DIR + pat))
        sn = as_rank(3, lambda: dsn._shard_stages(TAR_DIR + pat))
        samples_same = True
        for w in (0, 17):
            to = wds.DataPipeline(*so, wds.tarfile_to_samples(handler=wds.warn_and_continue))
            tn = wds.DataPipeline(*sn, wds.tarfile_to_samples(handler=wds.warn_and_continue))
            a = as_rank(3, lambda: as_worker(w, lambda: [(s["__key__"], s["__url__"]) for s in islice(iter(to), 40)]))
            b = as_rank(3, lambda: as_worker(w, lambda: [(s["__key__"], s["__url__"]) for s in islice(iter(tn), 40)]))
            samples_same &= a == b and len(a) == 40
        check("1d. pipeline", stages_same and samples_same, f"256 real shards: the same {len(po.pipeline)} pipeline stages; "
              "rank 3, workers 0 and 17: the same first 40 samples")
    finally:
        os.environ.pop("GPIC_SHARD_SEED", None)

    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
