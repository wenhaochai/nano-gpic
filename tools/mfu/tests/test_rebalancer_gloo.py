#!/usr/bin/env python3
"""CPU test of src/utils/rebalance.Rebalancer (MFU lever E) over 8 gloo processes: host_plan + exchange move every
sample to the rank the plan names, at its own batch position, with its caption tokens intact and in order, and y_pack
(cu, tok_b, pos, pidx, lens, Tp) describes the new packed batch. Run on a vis node: python test_rebalancer_gloo.py"""
import os, random, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PixelGen"))
W, B, D = 8, 32, 6


def sample_rows(r, i, n):
    """Token rows of sample (r, i): row j holds (r, i, j, ...)."""
    t = torch.zeros(n, D)
    t[:, 0], t[:, 1], t[:, 2] = r, i, torch.arange(n)
    return t


def worker(rank, steps, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    dist.init_process_group("gloo", rank=rank, world_size=W)
    from src.utils.rebalance import Rebalancer
    rb = Rebalancer(rebalance=True)
    bad = 0
    for step in range(steps):
        rng = random.Random(step * 100 + rank)
        T = rng.choice([16, 24, 48, 56, 168, 240])
        lens = [rng.randint(max(1, T - 16), T) for _ in range(B)]
        x = torch.zeros(B, 3, 4, 4)
        x[:, 0, 0, 0], x[:, 0, 0, 1] = rank, torch.arange(B).float()
        y = torch.cat([sample_rows(rank, i, n) for i, n in enumerate(lens)])
        types = [f"t{rank}_{i}" for i in range(B)]
        p = rb.host_plan(lens, types)
        x2, y2, pk = rb.exchange(x, y, p)
        src = x2[:, 0, 0, 0].long().tolist()
        pos = x2[:, 0, 0, 1].long().tolist()
        ok = pos == list(range(B))                                         # each sample at its own position
        lens2 = pk["lens"].tolist()
        exp = torch.cat([sample_rows(src[i], i, lens2[i]) for i in range(B)])
        from src.utils.rebalance import bucket
        S2 = exp.shape[0]
        ok &= y2.shape[0] == bucket(S2) and torch.equal(y2[:S2], exp) and bool((y2[S2:] == 0).all())   # tokens intact, in position order; zero dummy rows
        ok &= p["types"] == [f"t{src[i]}_{i}" for i in range(B)]
        cu = pk["cu"].tolist()
        ok &= cu[0] == 0 and cu[-1] == S2 and all(cu[i + 1] - cu[i] == lens2[i] for i in range(B))
        nd = bucket(S2) - S2
        ok &= torch.equal(pk["tok_b"], torch.cat([torch.repeat_interleave(torch.arange(B), torch.tensor(lens2)), torch.full((nd,), B)]))
        ok &= torch.equal(pk["pos"], torch.cat([torch.arange(n) for n in lens2] + [torch.zeros(nd, dtype=torch.long)]))
        Tp = pk["kmask"].shape[1]
        ok &= torch.equal(pk["pidx"], torch.where(pk["tok_b"] < B, pk["tok_b"] * Tp + pk["pos"], B * Tp)) and Tp >= max(lens2) and Tp % 8 == 0
        ok &= torch.equal(pk["kmask"], torch.arange(Tp)[None] < pk["lens"][:, None].long())
        ok &= torch.equal(pk["onehot"].sum(1), (pk["tok_b"] < B).long()) and torch.equal(pk["real"], pk["tok_b"] < B)
        # the step's samples, over all ranks, are the original ones
        got = [None] * W
        dist.all_gather_object(got, [(s, i, lens2[i]) for i, s in enumerate(src)])
        allo = [None] * W
        dist.all_gather_object(allo, lens)
        orig = sorted((r, i, allo[r][i]) for r in range(W) for i in range(B))
        ok &= sorted(t for g in got for t in g) == orig
        bad += not ok
    q.put((rank, bad))
    dist.destroy_process_group()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, steps, q)) for r in range(W)]
    for p_ in ps:
        p_.start()
    res = sorted(q.get() for _ in range(W))
    for p_ in ps:
        p_.join()
    nbad = sum(b for _, b in res)
    print(f"{'ok' if nbad == 0 else 'FAIL'}: {W} ranks x {steps} steps, failed rank-steps: {nbad}")
    sys.exit(1 if nbad else 0)
