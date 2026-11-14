# MoE line: infra notes (branch moe)

Plan: `tmp/ladder_prof/moe/PLAN.md` (owner approval 2026-11-13 22:18). Ledger: `project/gpic/logs/sweep_moe.yaml`.
Layer: `src/models/layers/moe.py`; balance and monitoring: `src/callbacks/moe_balance.py`; Triton sums (off by default):
`src/models/layers/moe_ops.py`.

## Step time and memory on 8 H100 (infra smoke w2680, 2026-11-14)

d12 = 1024 x (12+3), 128 images per GPU, 400 steps per arm, frozen worktree `gpic-wt/moe_smoke1` (moe b3ead47).
GPU time per step: median after step 20, first to last kernel of a step. X = expert parameters / dense FFN parameters,
G = active experts per token.

| arm | spec | mode | parameters | s/step | x dense | peak GiB |
|---|---|---|---|---|---|---|
| dense | dense | DDP | 0.38B | 0.1218 | 1.00 | 43.0 |
| ecb8 | ecb:8:2 (X 4) | DDP | 0.83B | 0.1633 | 1.34 | 57.4 |
| ecb16 | ecb:16:2 (X 8) | DDP | 1.43B | 0.1899 | 1.56 | 68.5 |
| eci16 | eci:16:2 | DDP | 1.43B | 0.1886 | 1.55 | 68.7 |
| tc16 | tc:16:2, capacity 1.5 | DDP | 1.43B | 0.2163 | 1.78 | 73.3 |
| ecb64g8 | ecb:64:8 (X 8, G 8) | DDP | 1.43B | 0.2203 | 1.81 | 77.9 |
| ecb16fs | ecb:16:2 | FSDP2 | 1.43B | 0.1623 | 1.33 | 49.7 |
| ecb32fs | ecb:32:2 (X 16) | FSDP2 | 2.49B | 0.1925 | 1.58 | 55.0 |

- Most of the MoE overhead scales with the parameter count, not with the routing. Under DDP: 44 ms per step per 1e9
  parameters (dense, ecb8, ecb16) plus about 22 ms for the routed layers. Under FSDP2: 28.5 ms per 1e9 (ecb16fs and
  ecb32fs have the same compute per token and differ only in expert parameters).
- FSDP2 is faster for every MoE shape here, needs about 19 GiB less, and compiles in 70 s instead of 327 s (one block
  graph for all blocks). Phase 1's MoE runs train under FSDP2.
- Granularity 8 costs 30 ms per step more than granularity 2 at the same parameters (DDP), and exceeds the 74 GiB mark
  under DDP.
- Token choice at capacity 1.5 balances with the bias rule over the first 400 steps: worst layer MaxVio 4.65 -> 1.07,
  dropped assignments 31.9% -> 8.8%.
- Resume (DDP, G0 e): all states restored; the 36 routing buffers are saved, set, measured and mirrored into the EMA
  slot; no statistics are saved. An FSDP2 checkpoint equals a DDP one in keys, shapes and dtypes (478 entries).
- A DDP arm compiles 4-5.5 min with the GPUs at 0%; a second compile of the same graph in the same job took 53 s
  (Inductor cache in the job's TMPDIR).

## One layer on one H100 (operator benchmark w2682, 2026-11-14)

`tools/moe/bench_moe_layer.py` and `bench_expert_gemm.py` at 32,768 tokens (128 images), forward + backward, bf16, compiled
(default mode) unless noted; worktree `gpic-wt/moe_bench1` (moe 4095805).

| d | config | dense ms | MoE ms | ratio | experts ms (TFLOPS) | gather | combine | route | NS dense / MoE ms |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | ecb n16 G2 | 4.45 | 6.18 | 1.39 | 5.02 (492) | 0.62 | 0.65 | 0.28 | 0.88 / 7.15 |
| 1024 | tc n16 G2, capacity 1.5 | 4.36 | 8.37 | 1.92 | 7.14 (346) | 0.78 | 0.85 | 0.38 | 0.87 / 7.14 |
| 1024 | eci n16 G2 | 4.36 | 6.12 | 1.40 | 5.03 (492) | 0.63 | 0.65 | 0.26 | 0.88 / 7.17 |
| 1024 | ecb n8 G2 | 4.45 | 5.93 | 1.33 | 4.69 (528) | 0.65 | 0.67 | 0.27 | 0.87 / 3.67 |
| 1024 | ecb n32 G2 | 4.52 | 6.63 | 1.47 | 5.57 (444) | 0.62 | 0.64 | 0.30 | 0.88 / 14.19 |
| 1024 | ecb n64 G8 | 4.52 | 8.92 | 1.98 | 5.57 (444) | 1.77 | 2.08 | 0.38 | 0.88 / 5.40 |
| 896 | ecb n16 G2 | 3.59 | 5.07 | 1.41 | 3.95 (479) | 0.55 | 0.57 | 0.27 | 0.70 / 5.87 |
| 896 | ecb n64 G8 | 3.61 | 7.98 | 2.21 | 4.99 (380) | 1.54 | 1.81 | 0.36 | 0.70 / 4.38 |

NS: Muon's Newton-Schulz of one layer's matrices on one GPU (a step splits it over 8 ranks).

- The routed layer adds 1.73 ms per layer at d12 ecb16, 20.8 ms per step, as the smoke's fit (about 22 ms).
- So the 40 ms that ecb16 adds under FSDP2 is about 21 ms routed layers, 9.4 ms Newton-Schulz and about 12 ms gradient
  reduction and optimizer passes for 1.06e9 parameters.
- The Triton segmented sums (`GPIC_MOE_KERNELS=1`) are slower than Inductor's fused index_add (1.43 vs 1.39 at G2, 2.10 vs
  1.98 at G8): they buy determinism, not speed.
- GEMM forms, eager, d12 n16 G2: the dense GEMMs 5.59 ms (442 TFLOPS); bmm with the transposed weight view 1.12 x; weights
  stored [n, d, 2h] 1.045 x; a loop over experts 2.5-13 x. Under torch.compile max-autotune the dense GEMMs reach 601 TFLOPS
  (0.74 x) and the bmm 538 TFLOPS (0.82 x): autotuning speeds up both sides, so it does not close the gap.
- The largest MoE-only kernels of a layer (d12 ecb16): the cat in the SwiGLU backward 0.45 ms, two index_add 0.26 ms each,
  the combine's backward reduction 0.20 ms, four casts 0.13 ms each. At G8 the two index_add take 1.03 ms each.

## The training step on 8 H100 (step profile w2686, 2026-11-14)

`tmp/ladder_prof/moe/prof_moe_step.sh`: d12, 128 images per GPU, torch.profiler steps 302-304 on all 8 ranks (GPU timeline),
views from who-ate-my-flops; full report in `runs/gpic-speed/moe_step_prof_1114_060842/report.md`.

| arm | GPU ms/step | x dense FSDP2 |
|---|---|---|
| dense, DDP | 121.2 | 1.03 |
| dense, FSDP2 | 118.0 | 1.00 |
| ecb16, FSDP2 | 161.4 | 1.37 |
| ecb16, DDP | 188.9 | 1.60 |
| ecb16, DDP, Triton sums | 188.3 | 1.60 |

The fair ratio is 1.37 x (both FSDP2). The MoE step is 39.4 ms busier than the dense one:

| source | +ms/step |
|---|---|
| glue of the gather and the combine (two fp32 index_add per layer, casts, zero fills; `moe.py` `_experts`) | about 16 |
| Newton-Schulz of the expert matrices | 6.7 |
| FSDP2 packing (chunk_cat of bf16 gradients into the fp32 reduce-scatter input, split copy-out) | 5.2 |
| Muon's pack and unpack copies around its all-to-all (`muon_fsdp.py`, per-unit loops) | 4.6 |
| optimizer element-wise passes on the shards | 2.4 |
| the model's GEMMs (expert bmm vs dense FFN) | 2.3 |

and it waits on the host: `MoEBalance.update` branched on GPU tensors (`if calls > 0`, `bool(moe_thr_set)`), 24 host
syncs a step, the first one waiting 75-90 ms for the backward; the optimizer and the next forward then ran host-bound.
moe 9215213 makes the update sync-free with the same values bit for bit (`tools/moe/tests/test_balance_nosync.py`); its
A/B is w2719. Under DDP the MoE is slower because every rank runs the whole optimizer (+13 ms element-wise, +17 ms
all-gather of the updates). The minimal traffic of the four glue passes is about 0.94 GB per layer (0.31 ms at 3 TB/s),
so fused bf16 sums could save up to about 10 ms a step; the Triton segmented sums of `moe_ops.py` do not (188.3 vs 188.9).

G0 f (step time <= 1.15 x dense at X 8, G 2) fails: 1.37 x under FSDP2 against the dense model under FSDP2.

## Three fixes, measured (A/B w2721, 2026-11-14)

`tmp/ladder_prof/moe/ab_moe_infra2.sh`: d12 ecb16 FSDP2, 400 steps per arm, two arms per code (repeats agree within 0.4 ms),
each code adding one change to the one before. GPU ms per step, median of steps 20-399:

| code | change | ms/step | vs old |
|---|---|---|---|
| old 4095805 | (phase 1's computation) | 161.8 | |
| nosync 9215213 | balance update without host syncs (same values bit for bit) | 159.1 | -2.7 (-1.7%) |
| v2 eac2681 | `GPIC_MOE_KERNELS=2`: token-order sums through the inverse slot map | 158.4 | -3.4 (-2.1%) |
| local aa8fced | `GPIC_MUON_LOCAL=1`: expert Muon parts owned where they live | 155.6 | -6.1 (-3.8%) |

- The profiled idle overstated the syncs' cost: the mean start-to-start time falls by 4.0 ms (2.4%), the slow steps (p90)
  by 9-11 ms.
- v2 does not pay: in the layer benchmark (fp32 input under autocast) it is slower than the torch path (6.56 vs 6.26 ms at
  d12 G2, 9.86 vs 8.85 at G8): the opaque gather is an ATen fp32 index_select (0.46 ms per layer) plus a cast that Inductor
  fuses in the torch path, and `_tsum_kernel` (16 tokens per program, a masked load per expert) takes 0.30 ms per layer, 3 x
  its bandwidth bound. Inductor's fused index_add is a strong baseline; the next step for the glue is a gather and scatter
  fused into the expert GEMMs.
- Newton-Schulz under torch.compile max-autotune on H100 (`tools/moe/bench_ns.py`): one rank's d12 G2 batch 10.49 -> 7.00 ms
  (369 -> 552 TFLOPS), d11 G2 7.59 -> 5.28, d12 G8 7.64 -> 4.58, with the same difference from fp32 (2e-2); on an A40 no gain.
  Switch `GPIC_MUON_NS_COMPILE` (`muon.py`).
- Phase 1 stays on 0381670 (the bit-for-bit change alone is under 3%); phase 2 takes nosync and local, and the
  Newton-Schulz switch after its own A/B. Fair ratio with nosync + local: 1.32 x (155.6 / 118.0).

## Newton-Schulz under max-autotune and the v3 operators (A/B w2722, 2026-11-14)

The layer benchmark's default mode (fp32 parameters and input under autocast) misled: an opaque gather read fp32 rows and
the layer cast its weights every call, which FSDP2 training does not. `bench_moe_layer.py --mp bf16` runs the layer as
FSDP2 trains it; there (A40, d12 ecb16, 16,384 tokens) v2's token sums are about 3 x faster than the fp32 index_add, and
v3 (`GPIC_MOE_KERNELS=3`: the same sums over a sort-free CSR, and a Triton gather) brings the glue kernels from 2.3 to
0.95 ms per call.

d12, FSDP2, GPU ms per step (median of steps 20-299), one worktree (f2ee279):

| code | ms/step | change |
|---|---|---|
| dense | 118.3 | |
| dense, Newton-Schulz max-autotune | 117.6 | -0.6% |
| MoE: nosync + local | 156.2 | |
| + Newton-Schulz max-autotune | 154.1 | -1.3% |
| + v3 operators | 151.5 | -1.7% |

MoE / dense: 1.320 -> 1.311 -> 1.289. Under max-autotune the first step takes about 3 min (one compile per batch shape);
a second launch with the same Inductor cache 7 s, so long runs should keep a persistent cache. A profile of the v3 arm
showed v3's CSR spending 0.22 ms a layer in a cumsum along the 16-wide expert dimension; moe 83a88d9 computes it as a
matmul with an upper triangle of ones (0.39 -> 0.03 ms on an A40, the same integers).

## Validation and the switch of phase 1 (w2725, 2026-11-14)

`tmp/ladder_prof/moe/val_p2_infra.sh`: d11 ecb16 FSDP2 with phase 1's baseline line, 3000 steps, two runs per code.

| code | GPU ms/step | mean loss, last 500 steps |
|---|---|---|
| 0381670 (phase 1 as queued) | 125.1, 125.1 | 0.07534, 0.07545 |
| 83a88d9 (nosync, local, NS max-autotune, v3) | 114.3, 114.4 | 0.07531, 0.07533 |

-8.6% per step. The largest gap between the 500-step mean losses of two runs of one code is 0.46% (0381670) and 0.26%
(83a88d9); between the codes' means 0.10%. Owner (popup): 「换成新 infra（推荐）」. Before any phase 1 run started, the 18
trainings' wq snapshots were replaced with `tmp/ladder_prof/moe/train_sweep_moe2.sh` (worktree `gpic-wt/moe_run2` at 83a88d9
and the three switches). Open: `GPIC_MOE_BMM_WT=1` (the experts' first GEMM with its gradients in their own layouts; bit
for bit with v3 on an A40) waits for an H100 A/B.

## Where the remaining time goes (w2722's profile against the dense one, 2026-11-14)

Kernel by kernel on rank 0 (d12, ecb:16:2, FSDP2; the v3 arm against the dense FSDP2 arm of w2686): 148.3 against 115.4 ms of
GPU kernels a step, +32.9 ms, none of it waiting (in a clean step the all-gathers overlap compute; 1.2 ms exposed).

| source | +ms/step | switch |
|---|---|---|
| Newton-Schulz of the expert matrices | 5.6 | (the algorithm) |
| FSDP2: bf16 gradients copied into the fp32 reduce-scatter input (chunk_cat), the all-gather's copy-out | 5.2 | `GPIC_MFU_FSDP_REDUCE=bf16` (changes the numerics) |
| the slot passes: two token sums, the combine's backward, the gather | 5.5 | v4 (one pass fewer) |
| Muon's element-wise passes and copies on the shards | 3.8 | `GPIC_MUON_FUSE=1` |
| the router: fp32 cast of the input, the TF32 GEMM, its backward | about 3.4 | `GPIC_MOE_ROUTER=1` |
| the CSR's cumsum | 2.7 | fixed in 83a88d9 |
| the weight gradient's layout copy | 1.1 | `GPIC_MOE_BMM_WT=1` |
| the model's GEMMs | 2.0 | |
| small kernels | about 3 | |

The slot kernels run near the memory bandwidth (an A40 profile at 128 experts with 16 active: the combine's backward at 82% of
the bandwidth, the token sums at about 75%).

## New switches (moe 108b4ad, 2e6ea1a; all off by default)

- `GPIC_MOE_ROUTER=1`: `gpic::moe_scores` computes the scores from the layer's input in its own dtype (TF32 products, as
  F.linear's GEMM on an H100; `GPIC_MOE_ROUTER_PREC=ieee` for IEEE products), with its own backward kernels: no fp32 copy of
  the input. Against F.linear the (token, expert) choices differ in 2e-4 to 8e-4 of the pairs (TF32 rounding).
- `GPIC_MOE_KERNELS=4` (v4): the gate multiplies the experts' hidden activations instead of their outputs (the same function):
  the combine is a plain sum over a token's slots, its backward a row gather, and the experts' outputs are not kept for the
  backward. A40 layer benchmark (`tools/moe/bench_moe_arms.py`, FSDP2's precision): -2.5% and -11% of the layer's peak memory
  at 128 experts with 16 active; +3% at 16 experts with 2 active (the gate's gradient becomes a reduction over the hidden side).
- `GPIC_MUON_FUSE=1`: Muon's momentum, Nesterov input and its cast in one launch, the weight decay inside the add
  (`src/optim/muon_add.py`). ATen's operations in ATen's order with explicit fused multiply-adds, so every value is the same
  bits (`tools/mfu/tests/test_muon_fuse.py`: 0 of 16.7M values differ; under FSDP2 on 2 GPUs, dense and MoE models, with and
  without `GPIC_MUON_LOCAL`, every parameter and momentum buffer equal). A plain `w * f + a * o` differs in a quarter of the
  values: the compiler contracts it the other way.
- `GPIC_MFU_FSDP_REDUCE=bf16`: FSDP2's reduce dtype; half the bytes of the reduce-scatter and of its input copy. Changes the
  gradient sum's numerics; refused with gradient accumulation (the micro-batches' sum would be kept in bf16).
- Tests: `tools/moe/tests/test_moe_ops4.py` (98 checks: the score operator against float64 with IEEE and TF32 products, the
  layer with v4 and the score operator against the torch path for every route and the granularity runs' shapes, evaluation,
  the choices under TF32), on an A40.

The H100 A/B is w2733 (`tmp/ladder_prof/moe/ab_moe_infra4.sh`, worktree `gpic-wt/moe_ab8` at 2e6ea1a).

## Granularity: the shapes' cost (shape smoke w2732, 2026-11-14)

`tmp/ladder_prof/moe/smoke_moe_grain.sh`: d11 = 896 x (11+2), 128 images per GPU, FSDP2, phase 1's code (83a88d9), 400 steps
per arm on 8 H100. GPU ms/step, x the dense arm, peak GiB:

| shape (experts : active) | expansion | ms/step | x dense | peak |
|---|---|---|---|---|
| dense, FSDP2, Newton-Schulz max-autotune | | 91.0 | 1.000 | 33.1 |
| 4 : 1 | 4 | 103.2 | 1.134 | 35.6 |
| 16 : 4 | 4 | 108.6 | 1.194 | 40.0 |
| 32 : 8 | 4 | 118.9 | 1.307 | 44.2 |
| 16 : 2 (phase 1's baseline, w2725) | 8 | 114.3 | 1.256 | |
| 128 : 16 | 8 | 144.0 | 1.582 | 55.6 |
| 16 : 1 | 16 | 129.4 | 1.422 | 40.5 |
| 64 : 4 | 16 | 135.6 | 1.490 | 44.9 |
| 128 : 8 | 16 | 147.1 | 1.616 | 49.4 |
| noise, 4 experts, capacity 2 | 4 | 137.8 | 1.514 | 42.2 |
| noise, 16 experts, capacity 2 | 16 | 163.4 | 1.796 | 47.4 |

Two costs add up. The experts' total parameters (expansion): at G = 1, 4 -> 16 adds 26 ms (Newton-Schulz grows with the
parameters times the smaller dimension, about 6 ms of it; the rest is FSDP2's traffic and the optimizer's passes). The
granularity at a fixed expansion: the slot tensors are [tokens x G, d], so every pass over them grows with G (8 -> 16 active at
expansion 8: +29.7 ms). The noise route pads every expert to twice its mean load (16 : 1 at capacity 2: +34 ms over ecb's
16 : 1, the same shapes otherwise).

A grouped GEMM that reads its rows through the slot table (no gathered copy; the gather's backward and the weight gradients
likewise) would remove up to four slot passes per layer, about 14 ms a step at 128 : 16. A Triton prototype on an A100
(scratch, not committed): correct, but 113-143 TFLOPS against cuBLAS's 181-220 for the same products, so it loses at 16 : 2
and about breaks even at 128 : 16 against v3's Triton gather. Not pursued without a TMA-class kernel.
