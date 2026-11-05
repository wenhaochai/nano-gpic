VERDICT: GO. No finding blocks a 10-minute timing job. The math and the measure are sound.

## Findings (none block)
1. AR_JiT.py:58-61, compile time. Dynamo unrolls 32 steps x 3 head blocks into one graph, about 100 small ops each step. The row count changes per draw (32 to 224). The first new shape recompiles with dynamic B; after that, no more recompiles. Estimate: 1 to 4 min per process for the flow variant. This is far below 1200 s. Fix: none needed. If first_batch_s > 600 s, use `mode="max-autotune-no-cudagraphs"` off (the default is already off).
2. time_head.py:43-50, loop runs first, then flow. The flow then runs on a warm GPU and a warm inductor cache. The bias is small next to a 2x Python-bound gap. Fix (optional): run the order flow, loop on 2 of the 4 seeds.
3. time_head.py:79, same_seed compares two different compiled graphs. Fusion order differs, so bf16 results can differ by a few ULP and grow over 64 steps. A nonzero max_abs is not a bug. Judge by mean_abs, not by equality. Test 19 checks exact equality only in eager mode (GPIC_AR_NO_COMPILE=1).

## What I checked
1. Math: HeadFlow.forward is the same expression as the loop, same grid, same `.float()`, same last-step rule. RNG for q is drawn outside, so the seed path is the same. Switch off: the old loop runs unchanged.
2. Grid: linspace is built inside forward, so Dynamo traces grid[j] as a tensor index, not a constant. No stale values across row counts.
3. Steps change: `_flow_steps != head_steps` builds a new HeadFlow and a new torch.compile, so a new graph. Test 19 covers this.
4. Autocast: same ops under the same autocast context as the loop; `.float()` after each head call in both.
5. Shapes: B=32 and the cosine schedule are fixed, so every batch has the same row counts. All shapes compile in the first batch. The 4 wall batches have no compile and no extra syncs.
6. Split: draw_patch is called from Python in _sample_random, so the class-level patch reaches it; net.compile() does not inline it. The syncs only run in the 2 split batches.
7. 4 processes: 48 CPUs, 4 OMP + 8 compile threads each. No CPU contention for a one-thread-per-GPU bottleneck.
8. Job: guard for old code and local changes; 25 min wall > 1200 s + 60 s kill; env.sh sets TMPDIR and proxy; the monitor is killed after wait.

## What I did not check
1. I did not run test 19 or any GPU code.
2. I did not check net.compile() internals beyond the call site.
3. I did not check gpu_health.py or the weights file.
