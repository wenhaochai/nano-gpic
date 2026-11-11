# The ladder speedrun: records, rules and speed work

Moved from the README on 2026-11-10. The README now covers only the ladder and how it is built.

**Current record: 419 H100-hours** (52.3 hours on one 8×H100 node), 37.3% MFU, for the first Muon ladder (README, part 1).

## Record history

| # | H100-hours | MFU | Description | Date | Code |
|---|---|---|---|---|---|
| 1 | 682 | 22.9% | The official PixelGen code: DDP, masked SDPA, crops resized on the CPU; AdamW, batch 256 | 2026-11-06 | [7ff7d5b](https://github.com/wenhaochai/nano-gpic/commit/7ff7d5b4) |
| 2 | 571 | 27.4% | FlashAttention-3 for the masked attention, the crop's resize on the GPU (bit for bit Pillow's), uint8 batches, FSDP2 from d14 | 2026-11-06 | [9b5017f](https://github.com/wenhaochai/nano-gpic/commit/9b5017f7) |
| 3 | 530 | 29.5% | Fused Triton kernels for residual + RMSNorm + adaLN modulation and for q/k RMSNorm + RoPE + the k/v concatenation; FSDP2 groups at d11 and d12 | 2026-11-07 | [b642496](https://github.com/wenhaochai/nano-gpic/commit/b642496f) |
| 4 | 438 | 35.7% | Recipe change: Muon (MuP scale) at batch 1024, run as 128 images per GPU with no accumulation; one caption-length bucket per step for all GPUs | 2026-11-07 | [7813166](https://github.com/wenhaochai/nano-gpic/commit/7813166c) |
| 5 | 428 | 36.5% | Patchify and unpatchify by reshape + permute at the FSDP2 sizes (d14 to d18), where they ran outside the compiled blocks and CUDA's unfold launched one im2col kernel per image (the same values bit for bit) | 2026-11-07 | [6560da0](https://github.com/wenhaochai/nano-gpic/commit/6560da0b) |
| 6 | **419** | **37.3%** | Muon's Newton-Schulz with its scalings and sums inside the GEMMs (the same polynomial, rounded once per GEMM) | 2026-11-07 | [0671b0e](https://github.com/wenhaochai/nano-gpic/commit/0671b0e0) |

H100-hours are training time: each model's steps times its mean measured step time times 8 GPUs, summed over the ladder; start-up compiles, checkpoints and evaluations are not counted. Records 1 to 3 run the official AdamW recipe at batch 256 and keep its math. Record 4 changes the recipe, at the same FLOPs per budget: Muon scores 14% to 24% lower FD than AdamW at every point tried. Record 5 keeps record 4's math; its step times at d14 to d18 are record 4's minus the saving measured on the same batches on paired GPU halves (12.5, 14.0 and 13.4 ms per step). Record 6 counts only the sizes measured the same way (4.4, 9.0 and 24.1 ms per step at d12, d16 and d18); its losses match record 5's on the same batches.

## Rules

1. The target is fixed: every model trained to its budget on GPIC Full, 8 H100s per model, the evaluation unchanged.
2. A speed change keeps the math: on the same batches its loss stays within the spread of repeated runs. A change that alters the numerics (FP8, a JPEG decoder, how captions are batched) counts only after an FD check at the official evaluation.
3. A step time is the mean over the second half of an arm of 1024 steps or more, on the production path, with every loss finite.

## Speed settings of the record

| model (width) | setting | ms per step (1024 images) | MFU |
|---|---|---|---|
| d9 (640) | DDP, 128 × 1 | 76.5 | 17.6% |
| d10 (768) | DDP, 128 × 1 | 83.6 | 25.4% |
| d11 (896) | DDP, 128 × 1 | 103.9 | 30.2% |
| d12 (1024) | DDP, 128 × 1 | 124.5 | 36.0% |
| d14 (1280) | FSDP2, 7 blocks per unit, 128 × 1, patchify by reshape | 198.7 | 40.5% |
| d16 (1536) | FSDP2, 128 × 1, patchify by reshape | 301.0 | 44.0% |
| d18 (1792) | FSDP2, 64 × 2, patchify by reshape | 439.2 | 45.8% |

## Speed work outside the record

- **NCCL.** NCCL 2.26.2 (the one torch 2.7.1 ships) returns the ranks' blocks of a single-node NVLS all-gather out of order at one message size, 576 KB per rank: the size of d16's final-layer shard, so every d16 checkpoint saved its final layer out of order while training ran correctly (NVIDIA/nccl #1906). Jobs now run NCCL 2.27.7 with NVLS on (checked correct, and 1-2% faster) or 2.26.2 with `NCCL_NVLS_ENABLE=0`.
- **FP8 GEMMs** (`GPIC_MFU_FP8=native3`): the image blocks' forward GEMMs and the MLP's input gradients in FP8, tensorwise scales from the last 16 steps' amax. In the ladder's setting it is 4.8% faster at d12 and 10.4% at d16; the casts to FP8 cost 17.5 and 36.1 ms per step there, against 10.8 and 60.4 ms saved in the GEMMs. FD passed at d9 (986.0, inside the bf16 seeds' 966.6 to 1016.6) but not at d14: at 1.4e19 FLOPs the held-out loss is 0.32% and FD 3.4% worse, about 13% more steps for the same loss, so the ladder stays bf16. At d16 in the recipe without the EMA, FP8 matches bf16 in held-out loss (0.06721 against 0.06721 and 0.06717). A first FP8 run's 10% worse FD came from a resume inside its cooldown, not from FP8 (README, step 8).
- **GPU JPEG decoding**: d9 and d10 are bound by the CPU's JPEG decoding. nvJPEG decodes about 84% of the training images in the training step, one step ahead, and the same exact resize runs on them: d9 runs 18% faster (77.2 to 63.1 ms per step), d10 the same, d11 and larger 3% slower. Its FD passed at d9 (984.5). The ladder keeps CPU decoding: d9 is off its critical path. Two bugs were found and fixed on the way: torchvision writes the decoded images on its own CUDA stream without waiting for the caller's, so in a running step it overwrote memory still in use; and a per-image copy loop held Python's lock.
- **Packed captions** are closed: their FD falls inside the shared bucket's seed spread, but they run 1-4% slower than one bucket per step.
- **Launch overhead** (branch `launch-cutfix-cap`): levers for `jobs/train_ladder.sh` (a gate beside main.py's start, a memo for the compiler's vector-set check, the thread count from the process start; off by default: every rank started at once, and no second copy of the weights into the EMA slot when no EMA is used), and the cut fix: the checkpoint writer starts with TERM blocked, so a cut right after a launch's first save leaves a whole checkpoint, and a failed launch leaves no process.

## Fixes

A torch 2.7.1 Inductor C++-wrapper template dropped a reference to `None` on every compiled call and crashed long runs (`src/utils/patch_bugs.py`). The shared caption bucket hung at the end of a data pass and dropped unfinished batches there (`src/data/shared_bucket_loader.py`, `src/data/dataset/gpic_dataset.py`). A cut by the job queue lost the steps since the last scheduled save: the training shell now waits on TERM for main.py's save (`save-on-term-cap`, `save-on-term-sched`). A checkpoint wrote the raw weights twice; it now writes them once (`ckpt-dedup-cap`, `ckpt-dedup-sched`).
