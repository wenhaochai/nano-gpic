# MFU of the PixelGen ladder training (branch `mfu`, from `ladder-full` 108afa1)

Owner, 2026-11-06: raise the training MFU at every ladder width, with every optimization that holds up, custom kernels included.

## Definition

MFU = training GFLOP per image (3 x forward, attention included, mean padded text length 51.5) x 256 images / 8 GPUs / step
time / 989.4 TFLOP/s. GFLOP per image: d640 104.1, d768 163.8, d896 242.1, d1024 346.9, d1280 622.4, d1536 1024.9, d1792 1554.8.
Step time: the mean over many steps (a sync spike is repaid by the next step), start to start, CUDA events.

## Tools

- `jobs/mfu/arm.sh`: one arm on the production path (jobs/train_ladder.sh settings, step timer, no checkpoints, no W&B).
- `tools/mfu/step_bench.py`: the same training step without Lightning and without the data path (synthetic batches with the
  measured text-length histogram), on one GPU or under DDP; `--record` gives loss and grad-norm curves for parity checks.
- `tools/mfu/attn_bench.py`: attention backends at the ladder's shapes, with timing and error against fp32.
- `tools/mfu/test_attn_parity.py`: a lever's attention through the whole denoiser against the production path, both vs fp32.
- `tools/mfu/trace_breakdown.py`, `tools/mfu/bench_summary.py`: kernel time per category; step-time table with MFU.

## Baseline (production path, 8 H100, node-j15g3, w1315, 2026-11-06)

| width | mean step | median | MFU | single GPU, no data path, no comm (w1316) | MFU there |
|---|---|---|---|---|---|
| d640 | 30.94 ms | 26.89 | 10.9% | 19.78 ms | 17.0% |
| d896 | - | - | - | 34.25 | 22.9% |
| d1024 | 57.47 | 56.28 | 19.5% | 43.46 | 25.8% |
| d1280 | 92.74 | 90.42 | 21.7% | 70.88 | 28.4% |
| d1536 | (136.7, w1313) | | (24.2%) | 107.03 | 31.0% |
| d1792 | 189.85 | 192.97 | 26.5% | | |

Kernel time per step (torch.profiler, 5 steps, rank 0; ms):

| width | GEMM | element-wise + norm | optimizer + EMA + clip | attention | NCCL (own stream) | compute idle |
|---|---|---|---|---|---|---|
| d640 | 5.6 | 7.3 | 2.1 | 3.3 | 1.8 | 11.7 (profiler on) |
| d1280 | 30.4 | 24.0 | 13.5 | 8.7 | 32.7 | 20.7 |
| d1792 | 75.5 | 49.5 | 35.1 | 18.4 | 48.1 | 14.7 |

Stage spans without the profiler (median, GPU ms): d1280 forward 20.1, backward 57.5, optimizer 11.1; d1792 forward 43.0,
backward 115.5, optimizer 27.8. The backward is 2.7-2.85 x the forward: it includes the wait for the gradient all-reduce.
NCCL picks NVLS + Simple for the 1000 MiB buckets; a 1 GB bucket takes 13-15 ms while the backward runs beside it and 5.6
ms alone (334 GB/s), so the all-reduce and the backward kernels slow each other.

## Best measured configuration per width (2026-11-07 00:1x; production path, 8 H100, the same batches)

| width | switches (jobs/train_ladder.sh) | step | MFU | levers off | job |
|---|---|---|---|---|---|
| d640 | A C D (defaults) | 24.4 ms | 13.8% | 26.3 ms, 12.8% | w1394 |
| d768 | A C D | 29.1 ms | 18.2% | 31.6 ms, 16.8% | w1396 |
| d896 | A B C D + L + G11 | 35.6 ms | 22.0% | 41.4 ms, 18.9% | w1422 |
| d1024 | A B C D + L + G6 + K + K2 | 40.2 ms | 27.9% | 52.1 ms, 21.5% | w1471 |
| d1280 | A B C D + L + G7 + K + K2 (T untested on top) | 61.0 ms | 33.0% | 83.9 ms, 24.0% | w1471 |
| d1536 | A B C D + K + K2 | 91.7 ms | 36.1% | 126.0 ms, 26.3% | w1471 |
| d1792 | A B C D + K + K2 | 131.0 ms | 38.4% | 178.9 ms, 28.1% | w1471 |

A: GPIC_MFU_ATTN=fa3; B: GPIC_MFU_FSDP=1 (ModelParallelStrategy); C: GPIC_MFU_GPU_RESIZE=1; D: GPIC_MFU_UINT8=1; L:
GPIC_MFU_LOCAL_OPT=1; G<k>: GPIC_MFU_FSDP_GROUP=k; K: GPIC_MFU_FUSED_NORM=1; K2: GPIC_MFU_FUSED_QK=1; T:
TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1 TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 (d1280 -1.9% before K); F
(GPIC_MFU_ASYNC_CKPT=1) for the checkpoints. GPU-h-weighted over the full ladder: 22.9% with levers off, 28.4% before
K + K2, 29.5% with this table. The running ladder (d1280-d1792 on f6713be with A+B+C+D+F, d640-d1024 on the frozen
code): 26.2%. Measured since: X (CUDA graphs) off; F8 (FP8) pays only at d1536 / d1792 and the new ladders stay bf16;
the per-rank caption length costs 6-28% (the shared bucket S fixes it for the new ladders); the new ladders run Muon at
global batch 1024 (sections below).

## Production path (Lightning, real GPIC Lite batches, 8 GPUs, the same batches in every arm), 2026-11-06

Mean step over 1024-1216 steps (jobs/mfu/v1_parity.sh, probe11, widths.sh; mean of every step after the first 100):

| width | levers off | A | A + C + D | A + B + C + D | A + C + D + E | job |
|---|---|---|---|---|---|---|
| d640 | 26.33 / 26.36 ms (12.8%) | 26.56 | 24.36 (13.8%) | 43.89 | 27.23 | w1389, w1394 |
| d768 | 31.61 (16.8%) | - | 29.09 (18.2%) | 46.90 | - | w1396 |
| d896 | 41.58 (18.8%) | - | 37.75 (20.7%) | 50.09 | - | w1396 |
| d1024 | 52.14 (21.5%) | 47.65 (23.5%) | 48.20 (23.3%) | 54.87 | - | w1394 |
| d1280 | 83.18 / 83.37 (24.2%) | 76.02 (26.5%) | - | 64.86 / 64.98 (31.0%) | 67.04 (A+B+C+D+E) | w1390, w1397 |
| d1536 | 126.00 (26.3%) | - | - | 95.93 (34.6%) | - | w1398 |
| d1792 | 178.94 (28.1%) | - | - | 136.16 (36.9%) | - | w1398 |

- B (FSDP2) needs a GPU-bound step: it adds host work per block (hooks, per-block graphs), which costs where the host binds
  (d640, d1024) and is hidden from d1280 on. Default: on from d1280.
- C + D pay where the data path's CPUs bind (d640: -7.6%); E costs more than it saves at d640 even with bucketed rows
  (its per-step gloo plan and two all_to_all calls add host work to a host-bound step). E stays off by default.
- d1536 and d1792 (w1398): A+B+C+D -23.9% at both. Floor (w1416, the same batches): the 500-step mean loss of three
  levers-off runs spans 0.65% at d1536 (0.168835 / 0.169265 / 0.169945) and 0.60% at d1792 (0.164804 / 0.165244 /
  0.165791); A+B+C+D (two runs each) 0.170557 / 0.170562 (d1536, above all three off runs) and 0.164507 / 0.164173 (d1792,
  below all three): the sign flips between widths, so chaos, not a bias of B; A+C+D at d1536 0.168990 (inside). Mean
  |loss diff| over the last 200 steps: off repeats 2.3-4.1e-3, lever arms 2.5-3.6e-3.
- Saves (lever F, w1391): a save step adds up to 15.3 s at d1536 and 21.9 s at d1792 (36.6 s and 57.8 s before).

Full ladder (ladder_full_run: C0 steps 534400 ... 36000, the same FLOPs per C0 at every width; trained to 1, 2, 4, 8, 8, 8,
8 x C0 from d640 to d1792), GPU-h-weighted MFU (total FLOPs / total time): levers off 22.9%, head defaults 27.4%; the
ladder as it runs (d640-d1024 on the frozen code, d1280-d1792 with A+B+C+D+F) 26.2%. GPU-h shares at the head: d640 5%, d768 8%, d896 13%, d1024 24%, d1280 18%, d1536 17%,
d1792 16%; d1024 is the largest share and the one with the most left (A+C+D 48.2 ms against 38.6 ms on one GPU).

B's checks besides the loss (2026-11-06): the EMA (tools/mfu/check_ema.py; on the tensors whose initial value is fixed,
the EMA's share of the denoiser's travel f and cos(ema - init, den - init)) at step 1600: f 0.0961 / 0.0967 / 0.0964, cos
0.9506 / 0.9504 / 0.9496 for 5cd3372's smoke, the head with DDP, the head with FSDP2 after a resume at step 600; at step
4800 f 0.2342 / 0.2353 / 0.2340. The FSDP2 last.ckpt holds Adam step 4800 for all 143 parameters (resumed at 600), the
same hyperparameters, exp_avg_sq per tensor between those of two DDP runs. A last.ckpt does not resume across strategies
(2 param groups under FSDP2, 3 under DDP). The real evaluation path (gmflow_run4 eval_gmflow.sh) loads the FSDP2 ms
checkpoint and samples; its sample check (std > 5) fails on any 1600-step checkpoint: the EMA there is ~90% initial weights.
Early per-step loss differences against levers off: every B arm 0 at steps 0-1, 2e-8 at step 2, 1e-6 at step 10, 2e-5 at
step 40, the same profile as DDP with FSDP2's per-block compile units (d640: 1e-8, 9e-7, 1e-5); A + C + D about 1e-8 over
the first 100 steps (d640: identical up to step 60). A math difference (gradient scale, unsynced replicated weights)
would show at steps 1-2.

Parity (the per-step global loss, mean of the 8 ranks): every arm, the levers-off repeat included, diverges from the
reference chaotically, from about 1e-8 at step 100 to about 1% by step 1000; what differs is the onset. A (FA3) and the
levers-off repeat stay below 1e-7 for 100 steps; arms with B start near 1e-5, and so does DDP with FSDP2's per-block
compile units and nothing else (probe11 d640_perblock: 1.67e-5 over the first 100 steps), so B's earlier onset comes from
the compile boundaries (the residual stream rounded to bf16 at every block boundary), not from FSDP2's math. Mean |loss
difference| over the last 200 steps against levers off: d640 off repeat 1.64e-3, A 1.67e-3, A+B+C+D 2.41e-3 (w1389);
A+C+D 6.5e-4 against the per-block control's 6.95e-4 (w1394, another seed of the chaos); d1024 A 1.98e-3, A+C+D 2.54e-3,
A+B+C+D 2.14e-3; d1280 A 2.87e-3, A+B+C+D 1.91e-3. Mean loss over the last 500 steps: d640 off 0.176508 / repeat
0.176503 / A 0.176514 / A+B+C+D 0.176917; d1024 off 0.181052 / A 0.181480 / A+C+D 0.180495 / A+B+C+D 0.180826; d1280 off
0.173608 / A 0.173773 / A+B+C+D 0.173339: the lever arms fall inside the scatter of the same-math arms.

## Levers (hypothesis, setup, expectation, result, decision)

### A. FlashAttention-3 for the masked attention (`GPIC_MFU_ATTN=fa3`)
- Hypothesis: SDPA cannot use its flash kernel with the key-padding mask, so production runs the sm80 memory-efficient kernel
  (45-70 TFLOP/s forward, 50-65 backward at these shapes, w1316). FA3 varlen with per-sample key counts computes the same
  attention without a mask and without the permutes and copies around it.
- Setup: `src/models/layers/fa3_op.py` (custom ops with fake kernels for torch.compile), HF kernels-hub build (owner OK
  2026-11-06). Alternatives measured in w1316: cuDNN with the mask (backward 30-50% faster than the current kernel under
  torch 2.7.1; torch 2.15 picks cuDNN by default and is 30-50% faster again), FlexAttention (1-2 TFLOP/s here: dropped).
- Expectation: attention 3-4x faster; d1280 about -6 ms per step.
- Result (w1321, step_bench on one GPU, SDPA / FA3 / SDPA on the same GPU, mean step):

| width | SDPA | FA3 | change | MFU (one GPU) |
|---|---|---|---|---|
| d640 | 19.75 ms | 17.54 | -11.2% | 17.1 -> 19.2% |
| d768 | 25.41 | 22.32 | -12.2% | 20.9 -> 23.7% |
| d896 | 34.18 | 30.17 | -11.7% | 23.0 -> 26.0% |
| d1024 | 43.55 | 38.61 | -11.3% | 25.8 -> 29.1% |
| d1280 | 70.68 | 63.74 | -9.8% | 28.5 -> 31.6% |
| d1536 | 107.35 | 97.50 | -9.2% | 30.9 -> 34.0% |
| d1792 | 154.82 | 142.01 | -8.3% | 32.5 -> 35.4% |

  The gain is larger than the attention kernels alone (FA3 also removes the permutes, copies and the cat around SDPA).
  Kernel speed at d1280 (H 20, T 56): FA3 forward 88 TFLOP/s and backward 81 against SDPA's 63 and 60; in the text
  blocks (T ~ 50 keys) SDPA is faster (0.23 vs 0.45 ms forward + backward), so the text blocks keep SDPA
  (GPIC_MFU_TEXT_ATTN=sdpa, the default from commit after w1321; w1321 ran FA3 in them).
- Correctness: tools/mfu/fa3_check.py (forward and backward against fp32, eager and compiled with a dynamic T, padded
  keys get exactly zero gradient); whole-model parity (tools/mfu/test_attn_parity.py, 4 configurations): FA3's error
  against fp32 equals SDPA's (ratio 0.97-1.01); 300 steps at d640 with the same batches: loss differs from the fp32-math
  run by 1.8e-6 (SDPA: 2.5e-6), and two FA3 runs differ from each other by 2.1e-6 (FA3's backward is not deterministic).
- Decision: keep. Reaches the full ladder only through the genmodel session (production path test: probe5).

### Where the 8-GPU step loses against one GPU (w1316, step_bench, synthetic batches, no data path)

| width | 1 GPU | 8 GPUs, one T for all ranks | 8 GPUs, own T per rank | production (real data) |
|---|---|---|---|---|
| d640 | 19.78 ms | 23.06 (+3.3 communication and DDP) | 28.96 (+5.9 rank imbalance) | 30.94 (+2.0 data and host) |
| d1280 | 70.88 | 82.11 (+11.2) | 88.16 (+6.1) | 92.74 (+4.6) |

Same-GPU repeats of step_bench agree within 0.5% (w1318). Loss floor for parity checks (w1318, d640, 300 steps, SDPA vs
SDPA's fp32 math backend, the same batches, t and noise): loss differs by 2.5e-6 relative at the end, grad norm by 1.6e-3.

### B. Communication and the optimizer (planned)
- Hypothesis: at d1280 and above, about 22-24% of the step is communication, the wait for the rank with the longest
  captions, and host work (8-GPU production vs one GPU); the optimizer and EMA cost 11 ms (d1280) and 28 + 7 ms (d1792).
  Sharding the optimizer and the EMA over the 8 ranks (reduce-scatter in DDP's buckets, AdamW and EMA on each rank's shard,
  all-gather of the updated parameters) divides their work by 8 with the same math.
- Setup: step_bench --fsdp (per-block fully_shard, bf16 compute params, fp32 reduce-scatter, RMSNorm weights and
  pos_embed replicated in fp32, fused AdamW and EMA on the shards); production path behind GPIC_MFU_FSDP=1
  (src/utils/fsdp_setup.py, ModelParallelStrategy via MFU_STRATEGY in jobs/mfu/arm.sh).
- Result (w1320, 8 GPUs, synthetic batches, own T per rank, SDPA attention, mean step):

| width | DDP (production) | DDP + AVG hook | DDP, 250 MiB buckets | FSDP2 | change | peak memory |
|---|---|---|---|---|---|---|
| d1280 | 87.99, 88.02 ms | 88.22 | 87.77 | 78.77 | -10.5% | 24.6 -> 15.9 GiB |
| d1792 | 186.49 | 187.57 | - | 157.93 | -15.3% | 52.9 -> 31.1 GiB |

  The AVG hook (no per-parameter divide kernels) and smaller buckets give nothing; FSDP2 does.
- Correctness: pending (loss parity against DDP, production path).

### B2. AdamW and EMA on the shards' local tensors, and blocks per FSDP2 unit (`GPIC_MFU_LOCAL_OPT=1`, `GPIC_MFU_FSDP_GROUP=k`)
- Hypothesis: B costs host time per step that a short step cannot hide: d640 +19.5 ms, d768 +17.8, d1024 +6.7 against
  DDP with A + C + D. Two sources: DTensor's dispatch of the optimizer and EMA ops, and FSDP2's hooks, one set per unit (16
  units at d1024).
- CPU (tools/mfu/dtensor_overhead.py, d1024's 129 sharded tensors, dispatch only): EMA lerp on DTensors 0.70 ms, on
  local tensors 0.13; fused AdamW on DTensors 2.78 ms, ShardLocalAdamW 1.06, plain 0.88 (a loaded vis node gave 2.1 / 14.8 /
  4.9 ms: the host's load scales it).
- Setup: src/utils/fsdp_setup.ShardLocalAdamW (torch 2.7's Adam.step with each DTensor's local tensor passed to
  torch.optim.adam.adam: the same fused kernel on the same data; the state keeps its DTensors, so checkpoints keep their
  layout); SimpleEMA lerps the local tensors, taken at every step. shard(group=k) makes k consecutive blocks one FSDP2
  unit (fully_shard of a module list: one all-gather, one reduce-scatter, one set of hooks).
- Correctness (CPU, 2 ranks, gloo): tools/mfu/tests/test_local_opt.py: ShardLocalAdamW = AdamW(fused) on DTensors bitwise
  over 6 steps (uneven and empty shards, a parameter without a gradient, weight decay 0 and 0.1, a group with its own lr),
  the same state_dict, and it continues bitwise from AdamW's state_dict; the local EMA lerp = the DTensor lerp.
  tools/mfu/tests/test_fsdp_group_cpu.py: a real FSDP2 step (forward, backward, replicated sync, clip, AdamW) with 1-block
  units and AdamW = 3-block units and ShardLocalAdamW, bitwise, and both = plain data parallel within 2e-7 (gradients) after
  one step. Note: gloo sums where FSDP2 asks for ReduceOp.AVG (the test translates it; NCCL averages). AdamW hides any
  uniform or per-tensor gradient scale (and the clip rarely acts at grad norm ~0.2 against 1.0), so a scale error never
  shows in the loss: check gradients or Adam's moments. The smoke's moments (exp_avg_sq FSDP2 / DDP: median 1.04 sharded,
  1.30 replicated, inside the DDP / DDP scatter) rule out a scale error in production.
- Expectation: B pays at d1024 (24% of the full ladder's GPU-h) and maybe d896.
- Result (w1405, w1406; production path, the same batches, 512 steps, mean of steps 101-511, ms):

| width | off | A+C+D | A+B+C+D | + L | + L + G3 | + L + G2 |
|---|---|---|---|---|---|---|
| d768 | 31.85 | 29.32 | - | - | 38.64 | - |
| d1024 | 52.28 | 48.43 (23.2%) | 54.59 | 52.92 | 42.85 (26.2%) | - |
| d1280 | 83.54 | - | 65.06 | 65.00 | - | 64.47 |

  At d1024 the hooks were the cost: 15 FSDP2 units -> 5 (G3) takes 10 ms off, and FSDP2 now beats DDP by 11.5%.
  L alone gives 1.7 ms. At d1280 (GPU-bound) G2 gives 0.6 ms. At d768 FSDP2 still loses by 9 ms.
- Parity (tools/mfu/parity_loss.py against A+B+C+D, its profiled repeat as the floor): d1024 L last-200 relative
  difference 7.1e-6, L+G3 2.9e-5 (floor 1.7e-5), the same mean loss over the last 500 steps to 6 digits; d1280 L+G2
  within the floor. Against levers off, every FSDP2 arm differs as before (the compile boundaries; see B's checks).
- Group sweep (w1418, 512 steps, ms): d1024 G3 42.51, G4 42.13, G6 41.34 (27.1%), A+C+D 48.50; d896 G3 39.76, G4 38.84,
  G6 37.21 (21.0%), A+C+D 37.89. Larger units win: fewer hooks, and FSDP2 still prefetches the next unit. Every
  L + G arm has A+B+C+D's 500-step mean loss to 6 digits.
- Decision: L + G6 at d1024 and d896 (the larger groups, 1 unit, and d768 / d1280 in w1422).

### G0. Inductor's autotuning of the element-wise and reduction kernels (Inductor flags; jobs/mfu/probe7.sh)
- Hypothesis: the element-wise and reduction kernels (RMSNorm with the residual add, modulation, gates, SwiGLU, RoPE, and
  in the backward the sums over tokens for the modulation gradients) take about as much GPU time as the GEMMs at d640 and
  0.8x at d1280 (w1315, w1382), more than their bytes need at 3 TB/s. Inductor picks their block sizes by heuristics,
  without measuring them.
- Setup: no code change. pw = TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1 (block sizes measured at compile time); cd = pw +
  TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1; ma = TORCHINDUCTOR_MAX_AUTOTUNE=1 (the GEMMs too: cuBLAS or Triton templates,
  epilogues fused into them) + coordinate descent. One GPU per task (step_bench, FA3), plain / tuned / plain on the same
  GPU, separate Inductor caches.
- Same math: pw and cd change block sizes, so the order of the fp32 sums in a reduction. ma also changes the GEMM
  kernels, and a fused epilogue can round once where the unfused graph rounds twice.
- Expectation: 5-15% off the element-wise time, 2-5% of the step.
- Result (w1412, one GPU, plain / tuned / plain, mean ms): pw gives nothing (d1792 141.6 / 142.2 / 142.0, d1280 63.7 /
  63.6 / 63.8, d1024 38.6 / 38.5 / 38.5, d640 17.4 / 17.2 / 17.1); cd: d1792 141.6 / 139.7 / 141.8 (-1.4%), d1280
  63.4 / 61.6 / 63.4 (-2.8%), d1024 38.6 / 37.6 / 38.4 (-2.2%); ma (GEMMs too) d1280 63.6 / 61.9: the same as cd, after a
  20-minute compile, so Triton GEMM templates win nothing over cuBLAS here. cd's compile adds 2-4 minutes, once per
  shape set (the Inductor cache keeps the tuned configurations).
- Per-kernel bandwidth (Inductor profile_bandwidth, d1280): the element-wise and reduction kernels of a step move
  about 39 GB (forward 14.8, backward 24.0); the worst are the backward's sums over tokens for the modulation gradients
  (triton_red_fused__to_copy_mul_sum_*, 0.9-1.3 TB/s alone) and many tiny kernels bound by their launch.
- Production (w1421, 1024 steps, mean ms): d1280 A+B+C+D 65.64 -> +T 64.40 (-1.9%); d1024 A+B+C+D+L+G3 42.92 -> 42.86
  (no gain: the FSDP2 step there is bound by the host); d640 A+C+D 24.60 -> 25.21 (within d640's noise; bound by the
  host and the data path). Parity: T's early divergence from levers off is A+B+C+D's exactly (first-100 relative
  difference 1.43e-4 at d1280 in both), the 500-step mean loss +0.12% at d1280 (inside the same-math spread of 0.3-0.65%;
  the 3x-floor rule flags it only because that job's floor pair happened to agree to 0.03%), PASS at d1024 and d640.
- Decision: T from d1280 up (GPU-bound widths), not below.

### H. The clip multiply, AdamW and the EMA in one kernel (`GPIC_MFU_FUSED_OPT=1`: clip + AdamW; `=2`: also the EMA)
- Hypothesis: under DDP every rank runs the clip's multiply (a read and a write of every gradient), torch's fused AdamW
  and the EMA lerp over all parameters: 2.3 ms of a 17.4 ms d640 step on one GPU (fused AdamW 1.26, EMA lerp 0.54, clip
  multiply 0.33, clip norm 0.18; probe9 trace w1382), at about 1.3 TB/s, so about 7.6 ms of d1024's 48 ms. torch's
  multi-tensor launches use 320 blocks of 512 threads per launch. One kernel with a block per 4096 elements of every
  tensor (address tables on the GPU), the clip coefficient applied in registers and the EMA read once with the new
  parameter moves 36 instead of 48 bytes per parameter at a higher bandwidth.
- Setup: src/utils/fused_adamw_ema.py (FusedAdamWEMA, a torch.optim.AdamW whose step runs the Triton kernel; the state
  stays AdamW's, so checkpoints and resumes are unchanged). configure_gradient_clipping computes the same norm and
  coefficient as torch.nn.utils.clip_grad_norm_ and hands the coefficient to the optimizer; with =2 SimpleEMA skips its
  lerp. DDP, and the local shards under FSDP2.
- Same math, bit for bit by design: torch 2.7's adam_math (double-precision moments and bias corrections, IEEE fp32
  division and square root) and lerp, operation for operation; nvcc's FMA contractions are matched by testing every
  placement against torch (tools/mfu/tests/test_fused_adamw_ema.py).
- Expectation: d1024 -4 to -5 ms (about -9%), d640 -1.5 ms; under FSDP2 (from d1280) about -1 ms.
- Result: bit for bit PASS (w1414: FMA_M=0, FMA_E=0, any FMA_V; 0 mismatching elements of param, exp_avg, exp_avg_sq and
  EMA over 6 steps at weight decay 0 and 0.1; every configuration of w1419 too). torch's kernels already ran at about
  2.4 TB/s, not 1.3 (the 1.3 came from a wrong parameter count). Time at d1024's 376 M parameters (norm + clip + AdamW
  + EMA): torch 8.0 ms; first kernel (scalar loads) 7.6 / 7.0 (H1 / H2); with 16-byte alignment hints and BLOCK 1024, 2-4
  warps (w1419) 6.2 / 5.6. Host side: the first version rebuilt its address tables whenever a gradient moved (every step
  on one GPU: d640 17.3 -> 19.3 / 23.5 ms); now only the gradient addresses are compared each step, through a pinned
  ring. Production (w1414, old kernel): d1024 A+C+D 48.27 -> H1 47.69 ms (-1.2%), H2 47.79; d640 no change (bound by the
  host and the data path); under FSDP2 nothing (d1536 96.03 -> 96.10, d1024 L+G3 42.51 -> 42.73), as the shards are 1/8.
  H2 also moves the EMA lerp from its side stream onto the critical path, so H1 is the better form.
- Decision: off by default. A DDP-width option (d640-d768) only; re-measure H1 with the aligned kernel there.

### B3. One flat parameter per FSDP2 unit (`GPIC_MFU_FSDP_FLAT=1`): not a general lever
- Observation: under FSDP2 the host binds part of the step at d1024-d1280. Per-stage medians (BENCH_STAGES, GPU / host
  ms): d1280 A+B+C+D forward 21.3 / 25.9, optimizer 1.7 / 7.8; d1024 A+B+C+D+L+G3 forward 14.0 / 17.1. In the profiled
  steps FSDP2's hooks hold about a third of the host's time (d1024 L+G3: pre_forward 9.3, post_backward_reduce 6.7,
  reshard 1.5, root callback 1.9 ms per profiled step), and the part that grouping cannot remove is per parameter:
  each step FSDP2 builds a new reduce-scatter output, makes one DTensor per sharded gradient and swaps every parameter
  between its sharded and unsharded tensor (about 200 parameters at d1024). Grouping took d1024 from 52.9 (G1) to 41.3
  ms (G6) and then stopped (G12 41.5).
- Design: the same math as B with flat buffers per unit and no per-parameter work per step: the module's parameters
  are bf16 views into one unsharded buffer per unit (all-gathered after the optimizer step, waited on per unit before
  its forward); gradients reduce-scattered in fp32 per unit when the unit's backward ends; the fp32 master weights, Adam
  moments and EMA as flat shards, updated by lever H's kernel on one tensor per unit, which also writes the bf16 shard
  for the next all-gather; RMSNorm weights and pos_embed replicated in fp32 as now. Checkpoints gather the shards into
  the same fp32 keys and AdamW states.
- Expectation: most of FSDP2's host work goes; d1024 and d1280 become GPU-bound: about -10% at d1024, -6% at d1280.
- Built (simpler than the design: FSDP2 keeps the master weights, AdamW and EMA shards; one flat fp32 parameter per unit,
  the original parameters its views). v1 read the views through per-instance property classes: every block got its
  own class, so Dynamo compiled one graph per block (14 graphs, 18 frames against 6 and 9) and past its 8-graph limit
  ran the rest eagerly: d1024 126.4 against 39.4 ms median (w1431, stopped after 2 of 6 arms). v2: a forward pre-hook
  (after FSDP2's, which unshards the unit) splits the flat parameter once per unit and sets the views as plain tensor
  attributes; the compiled blocks read them as ordinary inputs, so one graph serves every block of a class again (CPU
  test: graphs shared, none new over FSDP2's swaps, and under real FSDP2 with compiled blocks the same graph count and
  bitwise the same losses and gradients as per-parameter units).
- Result (w1456, production path, 2048 steps, mean step without -> with B3): d896 36.25 -> 34.39 ms (-5.1%), d1024
  41.15 -> 41.21 (none), d1280 63.65 -> 69.05 (+8.5%: 47 slow steps against 0, +1.5 GiB). Parity PASS at all three
  (first 100 steps 1e-8: the same math). Why so little: py-spy of a d1280 rank (next section) shows FSDP2's hook cost
  is per unit, not per parameter.
- Decision: off. Kept for d896 as a candidate if nothing better reaches it.

### Host time at the FSDP2 widths (py-spy, d1280 A+B+C+D+L+G7+B3, 8 s of one rank, w1456)
- The forward runs on the main thread, the backward on the autograd engine's thread. The main thread is busy for the
  whole forward (about 36 ms of a 64 ms step) and then waits for the backward (29% of its samples); the autograd thread
  is in Python for about 13 ms per step, so the backward keeps the GPU busy: the forward is the host-bound part.
- The forward's host time per step: kernel launches from the compiled blocks' wrappers 6.7 ms; FSDP2 pre-forward hooks
  7.5 ms (per unit: all-gather copy-in, the collective, copy-out, post-backward hook registration); FA3's path 3.4 ms
  (PyTorch's custom-op dispatch about 100 us and FA3's own C++ host code about 80 us per call); module calls, Dynamo
  frame entries and AOT wrappers about 5 ms; the eager parts of the step (noise and timestep sampling, embeddings, the
  loss) about 3 ms; B3's hook 1.9 ms. After the backward: the replicated gradients' all-reduce and the clip 2.1 ms.
- The step timer itself reads the CUDA memory statistics every step (about 1.3 ms at d1280). Production runs carry no
  step timer (train_ladder.sh adds it only with BENCH_OUT), so every measured arm's MFU is slightly low there.

### K. Residual add + RMSNorm + adaLN modulation as Triton kernels (`GPIC_MFU_FUSED_NORM=1`, src/models/layers/fused_norm.py)
- Owner, 2026-11-06 evening: the architecture will rarely change, so architecture-specific fused kernels are welcome.
- Each image block's two norm sites (norm1 + modulation; the first residual add + norm2 + modulation): one forward
  kernel each, and one backward kernel that reads dh, x_new, a and the upstream gradient once, writes dx and da, and
  keeps the column sums of shift, scale, gate and the norm weight per program (added afterwards in a fixed order).
  Inductor's backward of the same ops is several kernels, its column sums at about 1.2 TB/s (4.4 ms of d1280's 13.2 ms
  of backward Triton time, probe7). Registered with torch.library.triton_op: Inductor launches the kernels itself.
- Math: the model's ops in fp32 with its explicit casts kept (x_new and the normalized value rounded to bf16), the
  other intermediates unrounded as in Inductor's fused kernels, rsqrt as Inductor's (libdevice).
- Tests (tools/mfu/tests/test_fused_norm.py, CPU interpreter and GPU): the forward correctly rounded against fp64 (on
  the GPU up to the approximate rsqrt flipping a normalized value by one ulp, 0.0003-0.0007% of elements); every
  gradient at bf16 rounding (1.6-1.7e-3), below the eager ops' error (2.3-4.2e-3); the norm weight's 1e-7;
  deterministic. A tiny model: every gradient's error against fp32 within 1.07x of the plain path's.
- Kernel GPU time per site, forward + backward (w1468, profiler): below Inductor's at every width (site 1 -0.03 to
  -0.05 ms, site 2 -0.02 to -0.05 ms); the backward takes 32 rows per program above C = 1024.

### K2. q/k RMSNorm + RoPE + the k/v concatenation as Triton kernels (`GPIC_MFU_FUSED_QK=1`, src/models/layers/fused_qk.py)
- The image blocks' attention inputs on lever A's path: from qkv_x(x) and kv_y(y), one kernel per token kind writes q,
  k and v in FA3's layout (the per-head norm, the RoPE on interleaved pairs, image and caption keys and values
  concatenated); the backward reads dq, dk, dv and the projections once and writes the projections' gradients and the
  norm weights' column sums.
- Tests (tools/mfu/tests/test_fused_qk.py, CPU and GPU, the model's 16 and 28 heads): q, k correctly rounded, v exact,
  gradients at bf16 rounding and below the eager ops', the norm weights' 1e-7, deterministic. PASS on the GPU (w1468).
  A tiny model with K + K2: every gradient's error against fp32 within 1.05x of the plain path's.

### K and K2 on one GPU (w1468, step_bench, compiled, C++ launcher, FA3; mean step A / A+K / A+K+K2)
- d1024 38.86 / 37.78 / 36.73 ms (-5.5%); d1280 63.53 / 62.71 / 61.04 (-3.9%); d1536 98.75 / 95.83 / 93.27 (-5.5%);
  d1792 142.32 / 140.42 / 135.76 (-4.6%). Loss relative difference to A about 1e-6 over the first 100 steps (rounding;
  B's onset was 1e-5).

### K and K2 on the production path (w1471, 8 GPUs, 2048 steps; mean step without -> with K + K2)
- d1024 (+L+G6) 41.34 -> 40.21 ms (-2.7%), MFU 27.1% -> 27.9%; d1280 (+L+G7) 63.90 -> 61.01 (-4.5%), 31.5% -> 33.0%;
  d1536 95.80 -> 91.74 (-4.2%), 34.6% -> 36.1%; d1792 136.80 -> 130.96 (-4.3%), 36.8% -> 38.4%. d1024 gains least:
  its forward is host-bound, and K and K2 save GPU time.
- Loss over the last 500 steps, without / with: d1024 0.122221 / 0.121937, d1280 0.117095 / 0.116834, d1536 0.112095 /
  0.112119, d1792 0.110006 / 0.109274 (-0.67%, at the edge of the 0.60-0.65% spread of three same-math repeats there).
  The relative loss difference averages 0.5-2e-4 over the first 100 steps (A 1e-8, B 1e-5): K and K2 change the
  rounding inside every block. (The job's "PARITY PASS" lines test nothing: with two arms the second is the floor.)
- Decision: K + K2 on for runs not yet started (defaults unchanged until genmodel's OK).

### X. The compiled blocks as CUDA graphs (`GPIC_MFU_CUDAGRAPH=1`)
- Hypothesis: the FSDP2 widths' forward is bound by the host (previous section). Inductor's CUDA graphs (mode
  reduce-overhead) replay each block's kernels with one launch: the same kernels in the same order, so the same values.
- Setup: a graph keeps its parameters' addresses, so FSDP2 keeps each unit's unsharded storage between steps (it frees
  it after the backward and allocates it again, at a new address, before the next forward; the all-gather still
  writes the new values every step): one bf16 copy of the parameters per GPU (0.75 GB at d1024, 1.7 GB at d1792), and
  no change to the peak (every unit is unsharded at the end of the forward anyway). DDP's whole-model compile takes the
  same mode. A new caption length records new graphs once.
- Test (w1470): d1024 and d1280 with and without X, X with Inductor's Python launcher, d768 under DDP. Expectation:
  the forward becomes GPU-bound; bitwise the same losses.
- Result (w1470, w1474; the step's late part, steps >= 1024, median, without -> with graphs): d1024 39.7 -> 39.4 ms,
  d1280 62.9 -> 62.5, d896 34.4 -> 32.2 (-6.4%). torch 2.7 skips CUDA graphs with Inductor's C++ launcher ("skipping
  cudagraphs due to cpp wrapper enabled"), so X needs the Python launcher (letter Y); FA3's cached sequence offsets had
  to be made per call (a tensor cached during a warm-up lives in the graphs' pool outside every output, which cudagraph
  trees refuse). Every new caption length records every block again, and new lengths keep coming through 2048 steps:
  the mean is far worse (d896 36.5 -> 80.2 ms).
- Conclusion: the launches are not what binds d1024 and d1280 (py-spy without --native could not tell a launching
  host from a blocked one); X is off. What binds them is below (rank imbalance).

### F8. FP8 GEMMs in the blocks (`GPIC_MFU_FP8=tensorwise|rowwise`, src/utils/fp8.py; owner 2026-11-06 23:5x)
- torchao 0.11 float8 training (outside the shared env: envs/mfu-extra/torchao011) of the blocks' Linear layers (72 at
  d1024); adaLN, the embedders, the timestep MLP and the final layer stay bf16. Changes the numerics.
- Result (w1478, batch 256, FSDP2 + K + K2, 2048 steps, mean step bf16 -> FP8 tensorwise): d1024 39.91 -> 44.97 ms
  (+12.7%), d1280 61.06 -> 61.85 (+1.3%), d1536 91.69 -> 89.22 (-2.7%), d1792 131.43 -> 121.81 (-7.3%, MFU 38.3% ->
  41.3%); rowwise scaling at d1792 141.20 (+7.4%). The loss over the last 500 steps is within the same-math spread at
  every width (d1792 0.109857 bf16, 0.109769 tensorwise, 0.109492 rowwise). The casts (an amax and a cast per GEMM
  input, every micro-batch) cost more than the faster GEMMs below d1536.
- Decision (genmodel, 2026-11-07 01:5x): the new ladders stay bf16 at every width, so each ladder keeps one recipe; FP8
  for later runs is the owner's call.

### Rank imbalance: each rank pads to its own longest caption (w1482, step_bench, 8 GPUs, A + K + K2)
- Own T per rank against one T for all ranks, mean step: d640 (DDP) 25.13 vs 19.70 ms (+28%), d1024 (FSDP2) 48.58 vs
  45.73 (+6%), d1280 70.69 vs 61.60 (+15%), d1792 139.81 vs 130.97 (+7%). Per-sample padded lengths are bimodal
  (16-64: 90% of the samples; a second mode at 136-192: about 9%), so with 8 ranks about 57% of the steps hold one
  rank with T >= 72, and with 32 ranks about 97%. The largest lever left, and it grows with the GPU count.
- In the padded layout no deal of samples over the ranks helps: the rank that holds the longest caption pads to it.
  Options: packed captions (the same math; lever E's packed path cost more host time than it saved), or one length
  bucket per step for every rank (a change of which samples share a batch). The owner chose the second for the new
  ladders (popup 2026-11-07 01:3x; they start from scratch).

### S. The shared caption-length bucket (`data.train_dataset.init_args.shared_bucket: true`; tests: GPIC_SHARED_BUCKET=1)
- Buckets are padded lengths: 16 to 64 by 8, then 144, 176 and 320; a batch pads to its bucket's edge (mean padded T
  per micro-batch 50.3, against 51.5 for the per-rank grouping). Shares from the text cache's index (32 shards): 16:
  9.9%, 24: 27.4%, 32: 9.0%, 40: 10.5%, 48: 16.3%, 56: 13.5%, 64: 3.8%, 144: 2.7%, 176: 5.7%, 320: 1.3%.
- The workers (GPICWebDataset._group_shared) emit full batches of one bucket, greedily, holding fewer than one batch
  per bucket. On every rank SharedBucketLoader (src/data/shared_bucket_loader.py) keeps one pool of batches per bucket
  and takes each micro-batch from the bucket every rank holds most of: one gloo all-reduce (MIN) of 10 pool sizes per
  micro-batch, the same argmax on every rank. So every rank trains micro-batch k on the same bucket, also after any
  resume (the choice is made together each time; a resume restarts the pools empty on every rank), the pools stay
  bounded, the buckets come at their data share, nothing is dropped.
- A first version drew each micro-batch's bucket from one RNG shared by all ranks (no communication): its per-bucket
  queues are random walks against the draws and grow without bound (the CPU test dropped thousands of samples).
- CPU test (tools/mfu/tests/test_shared_bucket.py: 4 ranks x 8 workers, threads, the all-reduce as a barrier): the same
  bucket on every rank in all 3000 micro-batches, every sample in its bucket, shares within 7% of the data, pools at
  most 37 batches, nothing dropped or repeated.
- GPU check (w1576 on 59a13d8, w1591 on 060e69c, w1592; DDP, Muon, A C D K K2, 8 GPUs): with S every rank had the same
  T in every micro-batch (d640 and d1280 at 32 x 4: 2048 of 2048; d640 and d1024 at 128 x 1: 512 of 512), also after a
  stop and resume (d640 at 32 x 4: 512 of 512 after a stop at 128; at 128 x 1: 1024 of 1024 after a stop at 2048);
  without S 1287 of 2048.
- Speed at 32 x 4 (mean optimizer step, second half of 512): d640 92.5 ms without S, 106.3 with (+15%); d1280 247.0
  and 246.0 (0%). S lowers the slowest rank's GPU time per micro-batch (d640 26.0 -> 23.1 ms, d1280 62.2 -> 61.9) and
  raises the data wait (d640 1.7 -> 7.6 ms, d1280 2.1 -> 5.4): its all-reduce is a barrier for every host in every
  micro-batch, while DDP's no_sync lets the ranks run 3 of 4 micro-batches apart, so their imbalance averages over the
  4 micro-batches before they wait. At 128 x 1 every step syncs anyway; the control without S is w1596.

- A hang at the first data pass end (s8 w1722, w1724; s80 w1718; 2026-11-07 03:xx, my bug): one data pass is one epoch
  of the wrapped DataLoader, so the first rank to end a pass started a new Lightning epoch alone, and its new iterator's
  new_group waited for ranks that sat in the old group's all-reduce. Fix (genmodel, 31b2ffe): the loader restarts the
  wrapped iterator on StopIteration and never ends (the run is one epoch; the scheduler steps per step), and the gloo
  group is made once per loader. Test: tools/mfu/tests/test_shared_bucket_passes.py (8 gloo processes, different pass
  lengths per rank: PASS; ddd4b99 hangs).
- The workers dropped their unfinished bucket batches at every pass end, which costs the rare buckets most (an s8
  worker's pass held about 164 samples of bucket 320, about 64 of them still queued: about 39% of its long captions per
  pass; Lite about 12%, Full about 1.2%). 3338a8c keeps the queues on the worker's dataset copy, which persistent
  workers keep between epochs (tools/mfu/tests/test_group_shared_passes.py: 0 lost against 1832 of 37800).
- FD (genmodel's A/B, d640, Muon 3.4e-3, about 33.4k steps, FD with 50 Euler steps / held-out loss of the EMA):
  shared 128 x 1 1016.6 / 0.06857 (K + K2 off: 1006.3 / 0.06852); per-rank 128 x 1 982.6 / 0.06856; per-rank 32 x 4
  959.3 / 0.06854 (K + K2 off: 958.4); the lr search's code 949.2 / 0.06856. FD worsens as the buckets per optimizer
  step fall from 32 to 8 to 1, the kernels change nothing (pairs within 1%), and the held-out loss does not see it.
  Caption length is bimodal (90% at 16-64, 9% at 136-192), so a shared-bucket step at 128 x 1 is all short or all long
  captions. At Muon 6.8e-3 the shared bucket is only 0.8% above the search (973.0 against 965.2). The bucket's
  default waits for the owner. Costs: per-rank 128 x 1 is about +3% GPU-h (ladder MFU about 34.6%), per-rank 32 x 4
  about +18% (about 30.2%, partly estimated).
- Every process start (every resume) has a transient: the pools and the workers' queues start empty, and a worker emits
  its first batch of a rare bucket after 128 / share samples (176: about 2.2k, 320: about 9.8k), at about the same time
  on all 32 workers. In w1592 d640 the first 24 steps were all bucket 24, then 11 of 48, no T >= 144 before step 342,
  then a burst (14 long-caption steps in 29). Later the sequence is close to i.i.d. (same bucket as the step before:
  0.170 against 0.154).
- Test harness note: arm.sh gives every arm its own shard log (GPIC_SHARD_LOG_DIR=<out>/<tag>), so a resumed arm reads
  pass 0 again from its start (its bucket sequence repeats the first arm's). Production keeps one log per run.

### Muon (src/optim/muon.py, src/optim/muon_fsdp.py; owner 2026-11-07 00:5x, recipe from the popup at 01:0x)
- The question-5 reference (branch muon a403ed1: MuonAdamW for the blocks' matrices, AdamW for the rest; under DDP one
  owner rank per matrix part does the bf16 Newton-Schulz, one all_gather spreads the updates) with an update_scale
  switch: mup (the new ladders: -lr sqrt(d_out / d_in) msign, per split part) or moonlight (question 5's 0.2 sqrt(max)).
- ShardMuonAdamW (FSDP2): momentum and AdamW on the row shards; each part's Nesterov input goes to its owner in the
  Newton-Schulz dtype (all_to_all), the owner orthogonalizes it as the reference's owner does, the update rows come
  back (all_to_all). CPU test (2 ranks, gloo): bit for bit the reference under DDP (fp32 and bf16 Newton-Schulz, split
  on and off, both scales, weight decay), and a checkpoint round trip through the full state dicts gives the same bits.
- configs_h100/muon_ladder.yaml: split on, Nesterov 0.95, 5 steps, wd 0; AdamW 0.9 / 0.95 / 1e-8 / wd 0 at the width's
  lr; warmup 1000 optimizer steps, then flat; accumulation 4 (global batch 1024 on 8 GPUs); EMA 0.9996, once per
  optimizer step (the EMA now counts from the resumed step: with accumulation the first micro-batches after a resume
  end before any optimizer step).
- GPU (w1484): the reference tests pass on 1 and 2 GPUs; d640 Muon 256 optimizer steps (loss 0.2460 -> 0.0468); a stop
  at step 128 and its resume continue (0.0780 -> 0.0710 -> 0.0458, the uninterrupted run 0.0468). Cost per optimizer
  step (4 micro-batches): d640 91.5 against AdamW 88.8 ms (+3.0%), d1024 158.9 against 152.4 (+4.3%).
- 9928ffe was frozen for the lr search (muon_search_run 33dfdf4) and the narrow widths of the new ladders
  (muon_ladder_run 4607c61).

### Batch 1024: accumulation on DDP and FSDP2 (NoSyncModelParallelStrategy, widths.sh letter N)
- With accumulation, DDP syncs the gradients only in the last micro-batch (no_sync); Lightning blocks the sync only for
  DDP, so FSDP2 would all-gather and reduce-scatter in every micro-batch. NoSyncModelParallelStrategy does for FSDP2
  what no_sync does for DDP: before the last micro-batch, no reduce-scatter (the unsharded gradients add up in FSDP2's
  fp32 reduce dtype) and no reshard after the backward (the next micro-batch needs no all-gather).
- A d1024 micro-batch under DDP with accumulation costs 39.7 ms (w1484, A C D, no K), about FSDP2 + K + K2's 40.2 at
  batch 256. w1574 and w1530 meant to measure 32 x 4 per width, but their Muon overlay was an environment variable at
  submission, which wq's clean environment drops: they ran batch 256 with AdamW (512 rows, no MuonAdamW line). Their
  batch-256 numbers confirm the table above with K + K2 (d1280 FSDP2 + L + G7 60.8 ms, DDP 74.6; d1536 FSDP2 91.5, DDP
  113.3; d1792 FSDP2 131.5, DDP 163.3; d1024 FSDP2 + L + G6 39.9, DDP 46.5). A cold DDP start with K + K2 compiles the
  whole model (Dynamo splits it into 3 graphs at DDP's buckets): 240-282 s, 18-30 FX graph cache misses; FSDP2 compiles
  one graph per block class: 29-30 s, cache hits only. A cached start costs about 60-80 s under either.

### Batch 1024 per width (owner 2026-11-07: no accumulation where it fits; w1592 on 693e86c, w1596 the control)
- Global batch 1024 on 8 GPUs as micro-batch 128 x 1 (64 x 2 where 128 does not fit): the same math as 32 x 4 (the
  gradient is the mean over the same 1024 samples). Muon (MuP), the shared bucket (S), A C D K K2, 8 GPUs; mean
  optimizer step over the second half of each arm, MFU over 1024 images:

| width | setting | ms per step | MFU | peak reserved | without S | 32 x 4 (w1576) |
|---|---|---|---|---|---|---|
| d640 | DDP 128 x 1 | 76.5 | 17.6% | 19.3 GiB | 80.0 | 92.5 (no S) |
| d768 | DDP 128 x 1 | 83.6 | 25.4% | 25.4 GiB | | |
| d896 | DDP 128 x 1 | 103.9 | 30.2% | 32.4 GiB | | |
| d1024 | DDP 128 x 1 | 128.9 | 34.8% | 41.8 GiB | 132.7 | |
| d1280 | FSDP2 G7 128 x 1 | 211.2 | 38.1% | 52.5 GiB | 219.6 (DDP) | 247.0 (DDP, no S) |
| d1280 | DDP 128 x 1 | 213.5 | 37.7% | 60.7 GiB | | |
| d1536 | FSDP2 128 x 1 | 324.0 | 40.9% | 70.5 GiB | | |
| d1536 | FSDP2 64 x 2 | 333.9 | 39.7% | 37.8 GiB | | |
| d1792 | FSDP2 64 x 2 | 476.7 | 42.2% | 50.1 GiB | | |

- Ladder (GPU-h weights 1, 2, 4, 8, 8, 8, 8 from d640): 39 / sum(k / MFU) = 35.7% (batch 256 with AdamW, the best per
  width with K + K2: 29.5%; the start of this work: about 20%).
- 128 x 1 against 32 x 4, both DDP without S: d640 -13.5% (80.0 against 92.5), d1280 -11.1% (219.6 against 247.0). At batch 256 the same widths ran 24.4 x 4 = 97.6 and
  61.0 x 4 = 244 ms per 1024 images (AdamW).
- d1280: FSDP2 and DDP give the same losses on the same data (the first steps bit for bit, the last 200 0.09497 and
  0.09484); FSDP2 is 1.1% faster and needs 8 GiB less.
- d1536 at 128 x 1: the peak reserved is 70.5 GiB of 79.6 over 768 steps, which held no T = 320 batch; CompileWarmup ran
  a T = 320 batch of 128 before the timer reset the peak, inside the same pool, so T = 320 fits (about 7 GiB left with
  the optimizer states). d1792 at 128 does not fit (activations scale with d x blocks: about 88 GiB); activation
  checkpointing to fit it would recompute the forward (+25-33% compute), far more than 64 x 2 costs (d1536: +3.0%).
- S at 128 x 1: d640 -4.4%, d1024 -2.9%, d1280 -2.8% (DDP). It lowers the GPU time of the slowest rank (d640 77.9 ->
  64.7 ms max over ranks), but at d640 and d768 the data wait grows (d640 5.8 -> 36 ms per step, d768 25 ms) and the GPU
  idles. Even the rank that arrives last waits 16.7 ms (median), so the wait is in the DataLoader pulls, not only the
  barrier.
- S's pools are not bounded: each rank's batches of a bucket arrive independently, so the ranks' counts of a bucket
  drift apart like a random walk and only extra pulls realign them (8-rank CPU test, batch 32: the largest pool grows
  as about 4 sqrt(micro-batches / batch size), 35 -> 103 batches from 2000 to 20000 micro-batches; pool_min 80 only
  delays it). Estimate at 128 x 1: about 215 batches (about 14 GiB per rank) after an 8 h d640 segment; a resume
  empties the pools. Before 862d51e the steady state also took two all-reduces per micro-batch (a failed decision,
  then a pull); one pull before each decision makes it one, with the same pools.
- The workers are not slower with S (CPU, one process, 6 workers, 128 per batch: 1297 img/s with S, 1240 and 1301
  without), so the d640 wait is most likely the gloo all-reduce itself on a node whose 96 CPUs the 256 workers keep busy
  (two per micro-batch before 862d51e). From 862d51e rank 0 prints, every 1000 micro-batches, the largest pool, the
  decisions and pulls per micro-batch and the DataLoader and all-reduce waits, so the first ladder segments tell. A
  thread that takes the decisions ahead of the training loop would take the all-reduce off the step's path (not done).

### The none_dealloc crash (w1204 after 139 min, w1239; genmodel 2026-11-07): fixed in fdea752
- "Fatal Python error: none_dealloc" aborts Python 3.11 when None's refcount reaches zero (None is immortal only from
  3.12). torch 2.7.1's C++ launcher (TORCHINDUCTOR_CPP_WRAPPER=1) builds each compiled call's output list in
  pack_tensor_handle_list (torch/_inductor/codecache.py, CppWrapperCodeCache.extra_parse_arg) and stores Py_None for
  each null output with PyList_SET_ITEM, which steals a reference that was never taken: when the list dies, None loses
  one reference per None output (a backward graph returns None for every input without a gradient).
- Fix: src/utils/patch_bugs.py, imported first by main.py, rewrites the template to Py_NewRef(Py_None). Each cached
  wrapper module calls CppWrapperCodeCache.load_pybinding when it loads, which joins the template then, so FX graph
  cache hits get the fix too: every wrapper compiles once more under a new hash, and nothing else changes.
- Test (tools/mfu/tests/test_none_refcount.py, CPU): 1000 compiled forward + backward calls with one None output each;
  the unpatched launcher loses exactly 1000 references to None that the patched one keeps. torch's compiled path also
  takes about 4 references to None per call that it never releases, with the C++ or the Python launcher (eager 0; no
  gc-tracked object grows): harmless, and the reason a graph crashes only once its None outputs per call outnumber 4.
- Monitor: NoneRefcount (src/callbacks/none_refcount.py) in every ladder run writes rank 0's count to
  <run>/none_refcount.txt at each launch's first step and every 1000 optimizer steps.

### E with K2 on the packed layout (91b738e; NaN fix f3d78ad), and the bucketing options' gap (owner 2026-11-07)
- qk_prep_packed writes the packed caption keys and values straight into the image blocks' key layout with the fused
  norm (no per-block zero buffer and index_copy). Its first version left the caption rows past each caption
  uninitialised; FA3 masks those keys but loads the last V tile whole, and 0 x NaN is NaN: every EK arm of w1865, w1868,
  w1869 and w1870 trained to NaN from step 0 (d640, d1024) or 4 (d1280), and the FD arm w1871 too (moved to
  runs/gpic-speed/muon_bucket_ab_withdrawn). The speed summaries looked only at times. Fix: zero the caption part of k
  and v; the test now poisons the CUDA cache with NaN and runs FA3 end to end; arm.sh fails an arm whose loss is not
  finite. Re-timed on 8 GPUs: w1878. FD arm: w1875 (its first checkpoint, step 8352: all 288 tensors finite).
- The breakdown tool (tools/mfu/gap_breakdown.py) splits a profiled window into GEMM, attention, q/k preparation, norms,
  index/fill, optimizer, NCCL, memcpy, other, GPU idle, the NCCL straggler wait (a rank's NCCL time minus the smallest),
  the DataLoader wait and the shared bucket's waits. Inductor's fused_cat kernels compute (the largest is the SwiGLU
  backward) and count as other. Shared vs per-GPU buckets (valid arms, w1866): the GEMMs carry the padding (d1024 62.4
  vs 67.4 ms per step), the shared bucket waits on its own all-reduce at d640 (27.1 ms per micro-batch).
- Equal tokens per GPU for E (lever E's deal, R1 in the step, R2 on the host a step ahead; w1870): no gain (d640 R1
  77.0, R2 88.5 against 75.5 ms; d1024 R2 121.8 against 120.2), measured on the NaN code, but the deal's cost is host
  work and does not depend on the values. Dropped.

### F8b. Native FP8 (GPIC_MFU_FP8=native / native2, src/models/layers/fp8_native.py; owner 2026-11-07: 直接做b)
- Ceiling (w1879, tools/mfu/fp8_gemm_bench.py, one block's GEMMs at M = 128 x 256): FP8 (tensorwise scales) against
  bf16, forward + dgrad + wgrad: d640 1.33 / 2.11 ms (x1.59), d1024 2.88 / 4.86 (x1.68), d1280 4.13 / 7.50 (x1.82),
  d1792 8.04 / 14.33 (x1.78). One standalone cast pass costs 0.06-0.09 ms for a C-wide input and 0.32-0.60 ms for the
  4C-wide SwiGLU product: torchao's per-Linear casts (an amax pass, a cast pass, transposed copies, per GEMM) and our
  K kernels blocking Inductor's fusion of them explain why lever F8 was slower below d1536.
- torch 2.7's rowwise-scaled _scaled_mm is slow at our shapes (w1880, forward + dgrad): x0.68 at d640, x0.91 at d1024,
  x1.01 at d1280, x1.09-1.23 above, against x1.5-1.8 tensorwise. So the scales are tensorwise: an amax pass with an
  atomic max on the float's bits, then one cast pass (IEEE division: the casts match torch to the bit).
- Phase 1 (native): the image blocks' qkv_x, proj, w12, w3 forward GEMMs in FP8 (fast accumulation); the SwiGLU product
  cast by its own kernels (its bf16 output is never stored); the backward in bf16, the weight gradients of qkv_x, proj
  and w12 from their bf16 inputs. Phase 2 (native2): the MLP as one function with the dgrads of w3 and w12 in FP8 (full
  fp32 accumulation; the SwiGLU backward gathers dz's amax while it writes bf16 dz). Measured as pairs on the two GPU
  halves (bf16 or phase 1 on GPUs 0-3, FP8 or phase 2 on 4-7, the same batches): w1888, w1889.
- Next: the amax as a by-product of the producers (K's outputs, the SwiGLU), or delayed scaling, to fold the cast into
  one pass; FP8 weight gradients.

### E. Packed captions, the samples of each step dealt over the ranks (`GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1`)
- Hypothesis: each rank pads to its own longest caption, and the step waits for the slowest rank; long-caption batches
  (13%) make a step wait in about two thirds of the steps (5.9 ms at d640, 6.1 ms at d1280, w1316). A FLOP model of the
  measured length histogram puts the slowest rank at 1.11x the mean rank at every width; packing the captions and dealing
  the samples of each batch position over the ranks brings it to 0.99x (packing alone: 1.10x).
- Same math: a sample keeps its batch position (so its t and noise: the official seeding draws them alike on all ranks);
  each rank keeps 32 samples; loss and gradient are the same sums over the same 256 samples (up to fp32 order).
- Setup: model path with y_pack (text blocks and caption key projections on the real tokens; FA3 varlen in the text
  blocks; padded key layout with per-sample key counts in the image blocks); `src/utils/rebalance.py` (host plan through a
  gloo all_gather of lengths and caption types, device exchange by two all_to_all); collate packing; packed warm-up.
- Tests: exchange logic (tools/mfu/tests/test_rebalance_plan.py, 300 trials; test_rebalancer_gloo.py, 8 gloo ranks x 40
  steps): every sample once, at its position, tokens intact; after the plan the busiest rank holds 1.03x the mean caption
  tokens (worst 1.085).
- Result (w1323, partial; stopped by me at the DDP phase): whole-model parity of the packed path passes in 4
  configurations (error ratio to SDPA 0.98-1.02). Speed is worse, not better: one GPU, FA3 in both, mean step padded ->
  packed: d640 17.48 -> 36.95 ms, d1024 38.42 -> 42.45, d1280 64.20 -> 69.78, d1792 140.88 -> 143.90; 8 GPUs at d640
  26.14 -> 51.17. The packed steps take 35-45 ms with a few at 16 ms, at every caption length.
- Diagnosis so far: not recompilation (only two graphs, static then dynamic; a CPU test shows that Dynamo makes a changing
  int in a dict dynamic as well, so the shape-carried Tp of commit 18d8a69 is harmless but not the fix). The excess falls
  with width (+19.5 ms d640, +3-6 ms d1280-d1792), as a host sync per step would at a host-bound width. Candidate: the
  backward of the per-token modulation gather (index_add, bf16 atomics, through Inductor's proxy executor); commit b7ef645
  replaces the gather by a product with the one-hot token-to-sample matrix (the same forward values, an fp32-accumulated
  backward). probe9 (w1382): profiles of padded and packed at d640, the Python wrapper, eager, FA3 text attention.
- Decision for now: off by default in jobs/train_ladder.sh (genmodel, 2026-11-06), until fixed and measured.
- Where E loses (w1417): one GPU, packed against padded after the row bucketing: d1280 median 62.9 against 63.1 ms,
  d1024 38.0 against 37.6: the packed model path is nearly free now. In production, E costs +5.0 ms at d1280 under FSDP2
  (median 64.1 -> 69.1) and +0.8 ms at d1024 under DDP (47.6 -> 48.3). Per-stage times (BENCH_STAGES, d1280 A+B+C+D):
  the forward takes 25.9 ms of host time against 21.3 ms of GPU time (FSDP2's pre-forward hooks are about half of the
  host's forward), so the host binds the forward, and E's plan inside the step (a blocking gloo all_gather, the plan in
  Python) adds 5.2 ms of host time there.

### E2. E's deal on the host, one step ahead (`GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=2`, src/data/rebalance_loader.py)
- Hypothesis: the deal itself is cheap; E's cost is where it runs (in the step, on the host's critical path, and an
  all_to_all that lines up the ranks at the step's start).
- Setup: a thread per rank takes the next batch from the loader, writes its samples (lever C's source crops and resize
  tables, lever E's caption rows) to its shared-memory slot (all ranks of a run share one node), and joins one gloo
  all_gather of lengths and byte ranges, which is both the plan's input and the barrier; then it copies the samples
  src/utils/rebalance.plan gives this rank out of the 8 slots and rebuilds the batch in the packed collate's format.
  The step builds y_pack without any exchange. The same deal as E, so the same math (a sample keeps its position).
- Test (CPU, 8 gloo ranks, tools/mfu/tests/test_rebalance_loader.py): exact round trip of the batch format; every
  sample once per step, at its position, with its bytes; busiest rank's caption tokens 1.15-2.16x the mean -> 1.006-1.013x.
- First production test (w1420) was invalid: Lightning calls iter() on the training loader once to check that it is
  iterable and drops the result; the thread started there kept taking every other batch from the shared, persistent
  worker iterator, so the steps saw other samples than the reference arm (global loss 0.2% off at step 0) and the
  workers loaded twice as much (d640 waited 6.5 ms per step for data; d1280 +6.1, d1024 +10.4, d640 +8.4 ms). Fix: the
  thread starts at the first next() (6f73a7e); the CPU test now drops an iterator first. Retest: w1423 (1024 steps).

### D. uint8 crops (`GPIC_MFU_UINT8=1`)
- The workers keep the crop as uint8 HWC; the training step maps it with a 256-entry table that train_transform's own
  ToTensor + Normalize computed (no device arithmetic, so no question of how CUDA rounds x / 255). Bit-identical on 300 real
  crops (tools/mfu/tests/test_uint8_path.py). Saves the workers' ToTensor + Normalize (0.56 ms of 7.3 ms CPU per image on
  vis1) and 3/4 of the collate, pin and copy bytes. Matters only where the CPUs bind (d640, d768).

### F. Checkpoint writes in a separate process (`GPIC_MFU_ASYNC_CKPT=1`)
- Hypothesis: Lightning writes on rank 0 while every rank waits at the barrier: 57.8 s per last.ckpt at d1792, 36.6 s
  at d1536, 8 saves per C0, so about 6% and 4% of those runs' hours (the "saves included" MFU).
- Setup: `src/plugins/async_ckpt_io.py`: rank 0 copies the tensors into a shared, page-locked buffer (a few seconds), a
  writer process rebuilds and torch.saves the same dict, renames the file into place, and sets its mtime to the snapshot
  time (train_ladder.sh filters the shard logs by it). Two buffers (the ms checkpoint and last.ckpt of one step); teardown
  waits for every write. Same files, no math.
- Test: tools/mfu/tests/test_async_ckpt.py (CPU): three saves, tensors changed right after each save call; every file
  loads to exactly the saved state; mtime = snapshot time; no temporary file left. GPU timing: pending.

- Bug found 2026-11-06 (before any ladder run used it): Lightning 2.6.6's ModelParallelStrategy.teardown does not call
  checkpoint_io.teardown (DDP's does), so under lever B nothing waited for the final writes, and Python ends the daemon
  writer at exit: a final ms checkpoint or last.ckpt whose write outlasted the exit was lost (d640 won the race in w1399;
  d1792's 13.6 + 27 GB writes would not). Fix: the plugin also waits at interpreter exit (atexit, before multiprocessing
  ends its daemon children), never waits forever on a dead writer (exit 97 instead of a silent exit 0).
  tools/mfu/tests/test_async_ckpt_exit.py: old code (no exit wait): file missing, the race reproduced; new: file complete
  and equal, exit 0; writer killed with the write pending: exit 97 after the 10 s check, no hang.

### E and E2: conclusion (w1423, 1024 steps)
- With the bug fixed, E2 deals exactly E's batches (d1280: the same first-100 divergence profile as E, 3.28e-5), but
  d1280 A+B+C+D 65.51 -> +R 72.27 ms: the whole step-time distribution moves by 6-7 ms (median 63.7 -> 70.6), fast steps
  included, and the slow tail stays. So E's cost is the packed path's extra host work inside a forward the host already
  binds, not the plan's place; and the caption imbalance it removes is small: the slow-step tail (mean minus median) is
  1.6-2.4 ms at every width (1.5-6% of the step), part of which is not captions.
- Decision: E and E2 stay off; no further work on them while the FSDP2 step is host-bound.

### Review fixes (independent review of 5cd3372..e8bfff4, tmp/ladder_prof/review_mfu_head.md; mfu branch only, 6f73a7e)
- S1: LightningModel.state_dict joins the EMA stream first (ModelParallelStrategy gathers the state dict before the
  callbacks' on_save_checkpoint, so a save could read the EMA before the step's lerp finished).
- S3: the checkpoint writer ignores SIGTERM and SIGINT (a signal no longer cuts a write short and leaves a 13-27 GB
  .tmp<pid>), exits by itself when its parent is gone, and is killed on the exit-97 path; the first save of a launch
  removes .tmp<pid> files older than 30 minutes whose writer is dead.
- M1: the writer saves each tensor on a storage slice of the shared buffer instead of a copy (27 GB less host memory at
  d1792); torch.save writes the same bytes (checked), and empty tensors get their own storage.
- M3: the conditioners check on the device (torch._assert_async, no host sync) that every caption mask is
  left-aligned, as FA3's per-sample key counts assume. (Inductor removes asserts inside compiled graphs.)
- Tests: tools/mfu/tests/test_async_ckpt.py and test_async_ckpt_exit.py (old / new / dead) pass.

### C. Data path at the small widths (planned)
- d640 is bound by the node's CPUs (7.3 ms of worker CPU per image on vis1: decode 2.79, crop and resize 2.25, ToTensor
  0.35, Normalize 0.21, collate 0.25). Same-math candidates: uint8 batches normalized on the GPU (the same float32 ops);
  an exact GPU copy of Pillow's 8-bit bicubic resample (fixed-point coefficients, the same integer arithmetic).
