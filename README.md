# gpic-baseline: the GPIC ladder speedrun

This repository hosts the GPIC ladder speedrun: the fastest way to train the full compute ladder of PixelGen's JiT-T2I on [GPIC](https://gpic.stanford.edu) with 8 NVIDIA H100 GPUs. The ladder is seven widths, from 109M to 1.68B parameters, each trained to its budget on GPIC Full: 1, 2, 4, 8, 8, 8 and 8 × C0 from width 640 to 1792 (C0 = 1.43e19 FLOPs; 39 C0, about 5.6e20 FLOPs, in all). We run it to measure scaling laws for pixel-space text-to-image before testing new methods against them, so every hour saved is an hour of the ladder.

**Current record: 419 H100-hours** (52.3 hours on one 8×H100 node), 37.3% MFU.

This is a fork of the GPIC code release ([keshik6/gpic](https://github.com/keshik6/gpic); Chandrasegaran, Sargent et al., 2026; its README is [GPIC_README.md](GPIC_README.md)). Every run, number and figure is on the experiment page, [wenhaochai.com/exps/gpic-speed.html](https://wenhaochai.com/exps/gpic-speed.html); training logs are in [W&B reself/gpic-speed](https://wandb.ai/reself/gpic-speed).

*Updated 2026-11-07.*

## Record history

| # | H100-hours | MFU | Description | Date | Code |
|---|---|---|---|---|---|
| 1 | 682 | 22.9% | The official PixelGen code: DDP, masked SDPA, crops resized on the CPU; AdamW, batch 256 | 2026-11-06 | [7ff7d5b](https://github.com/wenhaochai/nano-gpic/commit/7ff7d5b4) |
| 2 | 571 | 27.4% | FlashAttention-3 for the masked attention, the crop's resize on the GPU (bit for bit Pillow's), uint8 batches, FSDP2 from width 1280 | 2026-11-06 | [9b5017f](https://github.com/wenhaochai/nano-gpic/commit/9b5017f7) |
| 3 | 530 | 29.5% | Fused Triton kernels for residual + RMSNorm + adaLN modulation and for q/k RMSNorm + RoPE + the k/v concatenation; FSDP2 groups at widths 896 and 1024 | 2026-11-07 | [b642496](https://github.com/wenhaochai/nano-gpic/commit/b642496f) |
| 4 | 438 | 35.7% | Recipe change: Muon (MuP scale) at batch 1024, run as 128 images per GPU with no accumulation; one caption-length bucket per step for all GPUs | 2026-11-07 | [7813166](https://github.com/wenhaochai/nano-gpic/commit/7813166c) |
| 5 | 428 | 36.5% | Patchify and unpatchify by reshape + permute at the FSDP2 widths, where they ran outside the compiled blocks and CUDA's unfold launched one im2col kernel per image (the same values bit for bit) | 2026-11-07 | [6560da0](https://github.com/wenhaochai/nano-gpic/commit/6560da0b) |
| 6 | **419** | **37.3%** | Muon's Newton-Schulz with its scalings and sums inside the GEMMs (the same polynomial, rounded once per GEMM) | 2026-11-07 | [LADDERPNS_LINK] |

H100-hours are training time: each width's steps times its mean measured step time times 8 GPUs, summed over the ladder; start-up compiles, checkpoints and evaluations are not counted. Records 1 to 3 run the official AdamW recipe at batch 256 and keep its math. Record 4 changes the recipe, at the same FLOPs per C0: Muon scores 14% to 24% lower FD than AdamW at every point tried. Record 5 keeps record 4's math; its step times at widths 1280 to 1792 are record 4's minus the saving measured on the same batches on paired GPU halves (12.5, 14.0 and 13.4 ms per step). Record 6 counts only the widths measured the same way (4.4, 9.0 and 24.1 ms per step at 1024, 1536 and 1792); its losses match record 5's on the same batches.

## In progress

- **FP8 GEMMs** (`GPIC_MFU_FP8=native2`): the image blocks' forward GEMMs and the MLP's input gradients in FP8, tensorwise scales. In the ladder's setting it is 4.8% faster at width 1024 and 10.4% at 1536. A profile shows where the rest goes: the GEMMs take 10.8 ms less per step at width 1024 and 60.4 ms less at 1536, but the casts to FP8 cost 17.5 and 36.1 ms. The next version (`native3`) casts each activation in one pass with a scale from its last 16 steps: 1-3% faster again at width 1024, 0.5% at 1280-1792; FP8 weight gradients (`native4`) are being timed. FP8 changes the numerics, so it counts only after that FD check.
- **GPU JPEG decoding**: widths 640 and 768 are bound by the CPU's JPEG decoding. nvJPEG decodes about 84% of the training images in the training step, one step ahead, and the same exact resize runs on them. On the same batches, width 640 runs 18% faster (77.2 to 63.1 ms per step). Two bugs were found and fixed on the way. (1) torchvision writes the decoded images on its own CUDA stream without waiting for the caller's stream, so in a running step it overwrote memory still in use. (2) A per-image copy loop held Python's lock. nvJPEG's pixels differ slightly from libjpeg-turbo's, so an FD check is running.

Packed captions are closed: their FD falls inside the shared bucket's seed spread, but they run 1-4% slower than one bucket per step.

## Rules

1. The target is fixed: every width trained to its budget on GPIC Full, 8 H100s per width, the evaluation unchanged.
2. A speed change keeps the math: on the same batches its loss stays within the spread of repeated runs. A change that alters the numerics (FP8, a JPEG decoder, how captions are batched) counts only after an FD check at the official evaluation.
3. A step time is the mean over the second half of an arm of 1024 steps or more, on the production path, with every loss finite.

## Running the current record

```bash
# usage: jobs/train_muon_ladder.sh <full|lite> <size> <d> <blocks> <text blocks> <C0 steps> <multiple of C0> <Adam lr> <Muon lr>
bash jobs/train_muon_ladder.sh full d640  640  9  2 133600 1 6.4e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d768  768  10 2 85024  2 5.0e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d896  896  11 2 57600  4 4.1e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d1024 1024 12 3 40192  8 3.3e-4 3.4e-3
bash jobs/train_muon_ladder.sh full d1280 1280 14 3 22400  8 2.4e-4 6.8e-3
bash jobs/train_muon_ladder.sh full d1536 1536 16 4 13600  8 1.8e-4 6.8e-3
bash jobs/train_muon_ladder.sh full d1792 1792 18 4 8992   8 1.4e-4 6.8e-3
```

Each line trains one width on one 8×H100 node, resumes from its last checkpoint, and carries the width's speed settings itself. Before the first run: GPIC Full (`tools/download_full.py`) and its Qwen3-1.7B caption features (`jobs/encode_captions.sh`, checked by `tools/check_text_cache.py`). The scripts keep absolute paths and a job-queue wrapper, so set them to yours.

| width | setting | ms per step (1024 images) | MFU |
|---|---|---|---|
| 640 | DDP, 128 × 1 | 76.5 | 17.6% |
| 768 | DDP, 128 × 1 | 83.6 | 25.4% |
| 896 | DDP, 128 × 1 | 103.9 | 30.2% |
| 1024 | DDP, 128 × 1 | 124.5 | 36.0% |
| 1280 | FSDP2, 7 blocks per unit, 128 × 1, patchify by reshape | 198.7 | 40.5% |
| 1536 | FSDP2, 128 × 1, patchify by reshape | 301.0 | 44.0% |
| 1792 | FSDP2, 64 × 2, patchify by reshape | 439.2 | 45.8% |

## The ladder

The goal:

1. Measure how image quality scales with model size, training compute and unique images in pixel-space text-to-image, in the units of language-model scaling studies: FLOP budgets, iso-FLOP curves, tokens per parameter, unique and total tokens.
2. Find how many passes over the data cost nothing, before fitting any optimum.
3. Only then test new methods, each against the ladder at the same compute.

- **Model.** PixelGen's JiT-T2I: x-prediction flow matching, 16-pixel patches (256 image tokens per 256 × 256 image), Qwen3-1.7B caption features. Widths 640 to 1792, from 109M to 1.68B parameters.
- **Recipe.** MiniT2I's settings (Adam, batch 1024, β2 0.95), turned into the MuP version of Muon by lr × 0.2√d ([Su, 2025](https://kexue.fm/archives/11416)): Muon at 3.4e-3 for the transformer matrices (6.8e-3 from width 1280: width 1280's own runs prefer it, width 1792's tie in FD and favour it in held-out loss, and width 1536 lies between them), Adam for the rest, 1000 warmup steps and a constant rate after them, EMA 0.9996 per step, no CFG.
- **Budgets.** One run per width, read at C0/4, C0/2, C0, 2C0, 4C0 and 8C0.
- **Data.** GPIC Full (101.1M images) and GPIC Lite (10.1M), plus repetition arms on the first 80 shards (1.01M images) and the first 8 (101K).
- **Scores.** FD-DINOv2 with the official evaluation (Euler at 50 steps, 50,000 test captions, one noise for all prompts, CFG 1), and the loss on 3072 held-out images at 16 noise levels with fixed noise.

## What we have so far

<img src="docs/readme/readme_all_points.png" width="560" alt="FD of every evaluation against training compute 6ND: the main-line ladder and the released model">

Every evaluation of every run, against training compute 6ND. Blue is the main line, the Muon ladder on GPIC Full, and the light-blue line follows the lowest FD of each of its budgets; red is the released GPIC model; both in the standard evaluation (Euler 50, the official noise, CFG 1). Grey is every other evaluation.

<img src="docs/readme/readme_muon_adamw.png" width="560" alt="FD at 1.4e19 FLOPs by parameters, Muon against AdamW">

1. **The Muon recipe beats the AdamW recipe at every point.** On GPIC Full at the same images, Muon scores 14% to 24% lower FD than the official AdamW recipe at all twelve points tried (widths 640 to 1024, 3.6e18 to 1.4e19 FLOPs). This compares recipe against recipe: the batch and the rates change with the optimizer.

<img src="docs/readme/readme_compute.png" width="560" alt="The lowest FD of each budget against training compute">

2. **The ladder on GPIC Full.** At 1.4e19 FLOPs the best widths are 896 and 1024 (FD 669.9 and 667.7), against 826.5 for the AdamW ladder on Lite. The released GPIC model (width 1536, AdamW, one pass over Full, 1.4e20 FLOPs) scores FD 472.4 at CFG 1. Widths 1280 to 1792 and the budgets above C0 are running.
3. **The learning rate falls slowly with run length.** In short runs the largest rate tried wins; at C0, 6.8e-3 beats 3.4e-3 by 5.9% at width 1280 (22.4k steps) and 3.1% at width 1024 (40.2k), and ties at width 640 (133.6k). Smaller rates lose at every width: at C0, 1.7e-3 is 2.4%, 2.1% and 11.2% worse than 3.4e-3 at widths 640, 768 and 1024, and 0.85e-3 is 11.6% worse at width 640.

<img src="docs/readme/readme_ema_lag.png" width="560" alt="EMA over raw held-out loss against the EMA's share of the initial weights">

4. **The EMA lags in short runs.** The EMA starts at the initial weights and keeps 0.9996^step of them. Above a 1% share its held-out loss is 4.5% to 590% above the raw weights'; below 0.2% it is 0.3% to 0.8% lower. Short runs and the first budgets of the widest models are read from the raw weights or need a later EMA; a sweep of the EMA decay is running.
5. **Repetition.** On the AdamW ladders, GPIC Lite costs nothing up to 17 passes over its 10.1M images: its held-out loss is within −0.12% to +0.07% of Full's at the same width and step. The 1.01M and 101K arms (135 to 1350 passes) are running.
6. **FD needs care.** Nine runs at width 640 whose held-out loss agrees to 0.07% span FD 949 to 1017. With one caption-length bucket per step, FD depends on the seed (966.6 to 1016.6) and averages 4% above runs with thirty-two buckets per step, which repeat within 0.3%. The ladder keeps the one-bucket setting, the fastest, and reads small FD gaps between single runs from the held-out loss.
7. **Fixes.** A torch 2.7.1 Inductor C++-wrapper template dropped a reference to `None` on every compiled call and crashed long runs (`src/utils/patch_bugs.py`). The shared caption bucket hung at the end of a data pass and dropped unfinished batches there (`src/data/shared_bucket_loader.py`, `src/data/dataset/gpic_dataset.py`).

## Where the code is

| Branch | Contents |
|---|---|
| `main` | the production line: the ladder (per-width speed settings, Muon also under FSDP2, the shared caption bucket), `jobs/train_muon_ladder.sh`, `configs_h100/muon_ladder.yaml`, and the job scripts of the learning-rate, EMA, caption-bucket and batch experiments |
| `mfu` | speed work: fused kernels, FSDP2, packed captions, FP8, GPU decoding; the log is `docs/mfu.md` |
| `muon-search2` | job scripts of the learning-rate, EMA, caption-bucket and batch experiments |
| `heldout-loss` | `tools/heldout_loss.py`: held-out and training loss at fixed noise, EMA and raw weights |
| `eval-par` | `jobs/evals_gmflow.sh --par`: one evaluation per GPU, eight side by side |
| `ladder` | the first ladder: the official AdamW recipe on GPIC Lite, no caption dropout |
| `muon` | the first optimizer ablation: Muon against AdamW at the official recipe |
| `gmflow` | a Gaussian mixture of the velocity at each step (K = 1, 8, 16) |
| `dist-step` | a distribution per step: notes and tests |
| `energy-head` | an energy-score head over the whole image |
| `ar` | autoregressive sampling of the image tokens |
| `implicit-nodrop` | a posterior sampler, and the point model without caption dropout |
