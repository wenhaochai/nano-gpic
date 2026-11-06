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

## Levers (hypothesis, setup, expectation, result, decision)

### A. FlashAttention-3 for the masked attention (`GPIC_MFU_ATTN=fa3`)
- Hypothesis: SDPA cannot use its flash kernel with the key-padding mask, so production runs the sm80 memory-efficient kernel
  (45-70 TFLOP/s forward, 50-65 backward at these shapes, w1316). FA3 varlen with per-sample key counts computes the same
  attention without a mask and without the permutes and copies around it.
- Setup: `src/models/layers/fa3_op.py` (custom ops with fake kernels for torch.compile), HF kernels-hub build (owner OK
  2026-11-06). Alternatives measured in w1316: cuDNN with the mask (backward 30-50% faster than the current kernel under
  torch 2.7.1; torch 2.15 picks cuDNN by default and is 30-50% faster again), FlexAttention (1-2 TFLOP/s here: dropped).
- Expectation: attention 3-4x faster; d1280 about -6 ms per step.
- Result: probe2 (w1317), pending.

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
- Pending: the DDP arms of w1316 (own T per rank vs one T for all ranks) split communication from rank imbalance.

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
  tokens (worst 1.085). GPU: probe4 (w1323), production path probe5 (w1324, w1325), pending.

### D. uint8 crops (`GPIC_MFU_UINT8=1`)
- The workers keep the crop as uint8 HWC; the training step maps it with a 256-entry table that train_transform's own
  ToTensor + Normalize computed (no device arithmetic, so no question of how CUDA rounds x / 255). Bit-identical on 300 real
  crops (tools/mfu/tests/test_uint8_path.py). Saves the workers' ToTensor + Normalize (0.56 ms of 7.3 ms CPU per image on
  vis1) and 3/4 of the collate, pin and copy bytes. Matters only where the CPUs bind (d640, d768).

### C. Data path at the small widths (planned)
- d640 is bound by the node's CPUs (7.3 ms of worker CPU per image on vis1: decode 2.79, crop and resize 2.25, ToTensor
  0.35, Normalize 0.21, collate 0.25). Same-math candidates: uint8 batches normalized on the GPU (the same float32 ops);
  an exact GPU copy of Pillow's 8-bit bicubic resample (fixed-point coefficients, the same integer arithmetic).
