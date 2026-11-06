#!/usr/bin/env python3
"""CPU test of step_bench.rebalance_plan and the exchange it drives: emulate all_to_all_single over 8 ranks and check that
every rank ends with one sample per position, each sample exactly once, its caption tokens intact and in position order,
and that the plan balances the caption tokens per rank."""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from step_bench import rebalance_plan, make_lengths, HIST

world, B = 8, 32
rng = random.Random(1)
worst = []
for trial in range(300):
    all_lens = []
    for r in range(world):
        T = rng.choices(list(HIST), list(HIST.values()))[0]
        all_lens.append(make_lengths(T, B, rng))
    # sample (r, i) = images "img r i", tokens ("tok", r, i, j)
    plans = [rebalance_plan(all_lens, me, world, B) for me in range(world)]
    sendbuf, tokbuf = [], []
    for me in range(world):
        send, in_s, out_s, inv, recv, new_lens, bal = plans[me]
        sendbuf.append([(me, i) for i in send])
        tokbuf.append([("tok", me, i, j) for i in send for j in range(all_lens[me][i])])
    for me in range(world):
        send, in_s, out_s, inv, recv, new_lens, bal = plans[me]
        got, got_t = [], []
        for src in range(world):
            s_send, s_in = plans[src][0], plans[src][1]
            off = sum(s_in[:me])
            got += sendbuf[src][off:off + s_in[me]]
            toff = sum(all_lens[src][p] for p in s_send[:off])
            tl = sum(all_lens[src][p] for p in s_send[off:off + s_in[me]])
            got_t += tokbuf[src][toff:toff + tl]
        assert len(got) == B and [g for g in got] == recv, (me, got[:3], recv[:3])
        x_new = [got[j] for j in inv]
        assert [i for (_, i) in x_new] == list(range(B))          # one sample per position, in order
        assert [all_lens[r][i] for (r, i) in x_new] == new_lens
        # token permutation as step_bench builds it
        starts, o = {}, 0
        for (r, p) in recv:
            starts[p] = o
            o += all_lens[r][p]
        tperm = [starts[p] + j for p in range(B) for j in range(new_lens[p])]
        y_new = [got_t[t] for t in tperm]
        exp = [("tok", r, i, j) for (r, i) in x_new for j in range(all_lens[r][i])]
        assert y_new == exp
    every = sorted(rc for me in range(world) for rc in plans[me][4])
    assert every == sorted((r, i) for r in range(world) for i in range(B))   # each sample once
    worst.append(plans[0][6])
print(f"ok: 300 trials; max/mean caption tokens per rank after the plan: mean {sum(worst)/len(worst):.4f}, worst {max(worst):.4f}")
