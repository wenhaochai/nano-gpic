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

### B3. FSDP2's per-parameter host work (planned)
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
