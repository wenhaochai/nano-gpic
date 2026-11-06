"""Full ladder (docs/ladder_plan.md section 12): the log of started shards and the resume of EpochShuffledShards, on CPU
without data. Run on a vis node:
  cd <worktree> && PYTHONPATH=baselines/PixelGen CUDA_VISIBLE_DEVICES= python tools/test_shard_resume.py
1. node split: rank r of 8 reads perm[r::8] of the pass's permutation; the 8 ranks read each shard once per pass
2. a stopped launch: 8 ranks x 4 workers each start part of their shards; each rank's log names each started shard
   once (also the one a worker was still reading), and only shards of that rank
3. resume: no worker of the resumed launch reads a logged shard, also not the ones in progress at the stop; logged +
   resumed = the rank's shards of the pass, each once; the next pass is whole and in the next permutation
4. every shard of a pass started before the stop: the resume starts in the next pass, with its started shards skipped
5. a log from another seed stops the resume with an assertion
6. only the training set logs: GPICWebDataset._shard_stages has the log stage for is_val False, not for is_val True
"""
import os
import sys
import tempfile
from itertools import islice

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "baselines", "PixelGen"))
from src.data.dataset.gpic_dataset import EpochShuffledShards, GPICWebDataset  # noqa: E402

WORLD, NW, SEED = 8, 4, 1762100000
URLS = [f"/d/gpic_train_{i:05d}.tar" for i in range(256)]   # 32 shards per rank, 8 per worker


def as_rank(r, f):
    old = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE")}
    os.environ.update(RANK=str(r), WORLD_SIZE=str(WORLD))
    try:
        return f()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def log_rows(d, r):
    p = os.path.join(d, f"shards_started_rank{r}.txt")
    return [line.split() for line in open(p)] if os.path.exists(p) else []


def main():
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{name}: {detail}  {'ok' if good else 'FAIL'}", flush=True)

    perm = {p: EpochShuffledShards(URLS, SEED)._perm(p) for p in (0, 1, 2)}

    # 1. node split
    seen, split_ok = [], True
    for r in range(WORLD):
        c = as_rank(r, lambda: EpochShuffledShards(URLS, SEED))
        mine = as_rank(r, lambda: [d["url"] for d in iter(c)])
        split_ok &= mine == perm[0][r::WORLD]
        seen += mine
    check("1. node split", split_ok and sorted(seen) == sorted(URLS), f"{WORLD} ranks read {len(seen)} of {len(URLS)} shards")

    # 2. a stopped launch, 3. its resume
    ds = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x")
    with tempfile.TemporaryDirectory() as d:
        in_progress, started = {}, {}
        for r in range(WORLD):
            # the workers' copies are forked from the rank's object before any worker writes: build them all first
            copies = [as_rank(r, lambda: EpochShuffledShards(URLS, SEED, d)) for _ in range(NW)]
            for w in range(NW):
                k = (r + w) % 3   # 0, 1 or 2 shards read to their end; worker (r, 3) of some ranks reads its whole list
                c = copies[w]

                def launch1():
                    ds._shard_log = c.log_path
                    g = ds._log_started(islice(iter(c), w, None, NW))   # split_by_worker, then the log stage
                    if w == NW - 1 and r % 2 == 0:
                        return [x["url"] for x in g], None        # to the end of the pass
                    got = [next(g)["url"] for _ in range(k + 1)]  # k shards read to their end, the next one in progress
                    return got, got[-1]
                got, cur = as_rank(r, launch1)
                started[(r, w)] = got
                in_progress[(r, w)] = cur
        log_ok, rank_ok, cur_ok = True, True, True
        for r in range(WORLD):
            rows = log_rows(d, r)
            logged = [u for _, p, u in rows]
            want = [u for w in range(NW) for u in started[(r, w)]]
            log_ok &= sorted(logged) == sorted(want) and len(set(logged)) == len(logged) and all(p == "0" for _, p, _ in rows)
            rank_ok &= set(logged) <= set(perm[0][r::WORLD])
            cur_ok &= ({in_progress[(r, w)] for w in range(NW)} - {None}) <= set(logged)
        n_logged = sum(len(log_rows(d, r)) for r in range(WORLD))
        check("2. stopped launch", log_ok and rank_ok and cur_ok,
              f"{n_logged} shards logged, each once, only by its rank, also those in progress")

        skip_ok, cover_ok, next_ok = True, True, True
        for r in range(WORLD):
            logged = {u for _, _, u in log_rows(d, r)}
            again = []
            nxt = []
            for w in range(NW):
                c2 = as_rank(r, lambda: EpochShuffledShards(URLS, SEED, d))
                again += as_rank(r, lambda: [x["url"] for x in islice(iter(c2), w, None, NW)])
                nxt += as_rank(r, lambda: [x["url"] for x in islice(iter(c2), w, None, NW)])   # the next epoch
            mine = perm[0][r::WORLD]
            skip_ok &= not set(again) & logged
            cover_ok &= sorted(list(logged) + again) == sorted(mine) and len(set(again)) == len(again)
            next_ok &= sorted(nxt) == sorted(perm[1][r::WORLD])
        check("3. resume", skip_ok and cover_ok and next_ok,
              "no logged shard read again (also not those in progress); logged + resumed = each rank's shards once; next pass whole")

    # 4. every shard of a pass started before the stop
    with tempfile.TemporaryDirectory() as d:
        r = 3
        mine0, mine1 = perm[0][r::WORLD], perm[1][r::WORLD]
        with open(os.path.join(d, f"shards_started_rank{r}.txt"), "w") as f:
            f.writelines(f"{100 + i}.000 0 {u}\n" for i, u in enumerate(mine0))
        c = as_rank(r, lambda: EpochShuffledShards(URLS, SEED, d))
        whole = (c.passes, c.skip) == (1, set())
        with open(os.path.join(d, f"shards_started_rank{r}.txt"), "a") as f:
            f.writelines(f"{200 + i}.000 1 {u}\n" for i, u in enumerate(mine1[:2]))
        c = as_rank(r, lambda: EpochShuffledShards(URLS, SEED, d))
        part = (c.passes, c.skip) == (1, set(mine1[:2]))
        got = as_rank(r, lambda: [x["url"] for x in iter(c)])
        check("4. pass started before the stop", whole and part and got == mine1[2:],
              "all of pass 0 logged: resume in pass 1; 2 of pass 1 logged: those 2 skipped")

    # 5. a log from another seed
    with tempfile.TemporaryDirectory() as d:
        r = 1
        other = EpochShuffledShards(URLS, SEED + 1)._perm(0)
        foreign = next(u for u in other[r::WORLD] if u not in perm[0][r::WORLD])
        with open(os.path.join(d, f"shards_started_rank{r}.txt"), "w") as f:
            f.write(f"100.000 0 {foreign}\n")
        try:
            as_rank(r, lambda: EpochShuffledShards(URLS, SEED, d))
            raised = False
        except AssertionError:
            raised = True
        check("5. foreign log", raised, "a shard of another seed's split stops the resume")

    # 6. only the training set logs
    with tempfile.TemporaryDirectory() as d:
        old = {k: os.environ.get(k) for k in ("GPIC_SHARD_SEED", "GPIC_SHARD_LOG_DIR")}
        os.environ.update(GPIC_SHARD_SEED=str(SEED), GPIC_SHARD_LOG_DIR=d)
        try:
            spec = "/d/gpic_train_{00000..00063}.tar"
            tr = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x")
            va = GPICWebDataset(tar_dir="/nonexistent/", tar_pattern="x", is_val=True)
            st_tr, st_va = tr._shard_stages(spec), va._shard_stages(spec)
            good = (len(st_tr) == 3 and st_tr[2] == tr._log_started and tr._shard_log == os.path.join(d, "shards_started_rank0.txt")
                    and len(st_va) == 2 and va._shard_log is None and st_va[0].log_path is None)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        check("6. training set only", good, "the training set has the log stage, the validation set has none")

    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
