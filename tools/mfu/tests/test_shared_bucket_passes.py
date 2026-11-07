#!/usr/bin/env python3
"""SharedBucketLoader across data passes (src/data/shared_bucket_loader.py; the hang of 2026-11-07, w1718 w1722 w1724).
8 gloo ranks on the CPU, one process each, with the real gloo group of the loader. Each rank has a fake training loader
whose every iter() is one data pass (the persistent DataLoader workers give one pass per epoch), with a different number
of batches per rank and pass, so the ranks end their passes at different micro-batches. Each rank runs Lightning's loop
shape: next() until max_steps, a new iterator when one ends (a new epoch), and one all-reduce on the default group per
step (the gradient all-reduce). Checks: no hang (the parent kills the ranks after the timeout); every rank trains the
same bucket at every micro-batch; no batch twice and none lost (pulled = trained + still pooled); one epoch on every
rank; every rank crossed at least 2 pass ends.
usage: test_shared_bucket_passes.py [--code <PixelGen dir>] [--timeout <s>]
--code: the code to test (default: this worktree's); the code before the fix (ddd4b99) hangs at the first pass end."""
import argparse, json, os, random, socket, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = (16, 24, 32, 40, 48, 56, 64, 144, 176, 320)
SHARE = (.10, .27, .09, .11, .16, .13, .05, .04, .03, .02)   # about the measured bucket shares


class PassLoader:
    """One data pass per iter(); 120 + 9 r + U[0, 40) batches in pass p of rank r, each of one bucket."""

    def __init__(self, rank):
        self.rank, self.passes, self.yielded = rank, 0, []

    def __iter__(self):
        import torch
        p, self.passes = self.passes, self.passes + 1
        rng = random.Random(7919 * self.rank + p)
        for i in range(120 + 9 * self.rank + rng.randrange(40)):
            T = rng.choices(EDGES, weights=SHARE)[0]
            self.yielded.append([self.rank, p, i])
            yield None, (self.rank, p, i), {"text_mask": torch.zeros(1, T, dtype=torch.bool)}


def child(r, world, init, code, steps, out):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    sys.path.insert(0, code)
    from src.data.shared_bucket_loader import SharedBucketLoader
    dist.init_process_group("gloo", init_method=init, rank=r, world_size=world)
    loader = PassLoader(r)
    sb = SharedBucketLoader(loader, EDGES)
    it, step, epochs, trained = iter(sb), 0, 1, []
    while step < steps:
        try:
            b = next(it)
        except StopIteration:   # Lightning: the epoch ends on this rank, and the next epoch starts a new iterator
            epochs += 1
            it = iter(sb)
            continue
        trained.append([list(b[1]), int(b[2]["text_mask"].shape[1])])
        t = torch.ones(1)
        dist.all_reduce(t)   # the step's gradient all-reduce
        step += 1
    pooled = [list(x[1]) for q in it.pools for x in q]
    with open(out, "w") as f:
        json.dump(dict(rank=r, trained=trained, pooled=pooled, pulled=loader.yielded, epochs=epochs,
                       passes_started=loader.passes, passes_ended=getattr(sb, "passes", None)), f)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=os.path.abspath(os.path.join(HERE, "..", "..", "..", "baselines", "PixelGen")))
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    import multiprocessing as mp
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    tmp = tempfile.mkdtemp(prefix="sb_passes_")
    outs = [os.path.join(tmp, f"rank{r}.json") for r in range(a.world)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=child, args=(r, a.world, f"tcp://127.0.0.1:{port}", a.code, a.steps, outs[r]))
             for r in range(a.world)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(max(0.0, t0 + a.timeout - time.time()))
    alive = [r for r, p in enumerate(procs) if p.is_alive()]
    if alive:
        for p in procs:
            p.kill()
        done = [r for r in range(a.world) if os.path.exists(outs[r])]
        print(f"HANG: ranks {alive} still running after {a.timeout:.0f} s (finished: {done}); code {a.code}")
        sys.exit(1)
    bad = [(r, p.exitcode) for r, p in enumerate(procs) if p.exitcode != 0]
    assert not bad, f"ranks exited with errors: {bad}"
    res = [json.load(open(o)) for o in outs]
    seqs = [[T for _, T in x["trained"]] for x in res]
    assert all(len(q) == a.steps for q in seqs), [len(q) for q in seqs]
    assert all(q == seqs[0] for q in seqs), "the ranks trained different buckets"
    for x in res:
        ids = [tuple(i) for i, _ in x["trained"]]
        pooled = [tuple(i) for i in x["pooled"]]
        pulled = [tuple(i) for i in x["pulled"]]
        assert len(set(ids)) == len(ids), f"rank {x['rank']}: a batch trained twice"
        assert set(ids).isdisjoint(pooled), f"rank {x['rank']}: a trained batch still pooled"
        assert sorted(ids + pooled) == sorted(pulled), (f"rank {x['rank']}: {len(pulled)} pulled, {len(ids)} trained, "
                                                        f"{len(pooled)} pooled: batches lost")
        assert x["epochs"] == 1, f"rank {x['rank']}: {x['epochs']} epochs (a pass end ended the epoch)"
        assert x["passes_ended"] is not None and x["passes_ended"] >= 2, f"rank {x['rank']}: {x['passes_ended']} pass ends"
        assert x["passes_started"] == x["passes_ended"] + 1
    ends = [x["passes_ended"] for x in res]
    pool_end = [len(x["pooled"]) for x in res]
    print(f"PASS: {a.world} ranks x {a.steps} micro-batches in {time.time() - t0:.1f} s, one epoch each, the same bucket on "
          f"every rank at every micro-batch, nothing lost; pass ends per rank {ends}; batches pooled at the end {pool_end}")


if __name__ == "__main__":
    main()
