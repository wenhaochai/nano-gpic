# Text encoder outputs read from disk

**Finding.** Step time across the five ladder sizes is 41 ms + 0.14 ms per GFLOP/img (w920), so 58% of the sweep's
training hours is a part the denoiser's FLOPs do not explain. w953 located it on 8 H100s (runs/gpic-speed/ladder_prof_1102_2209):

1. d640 stages, GPU ms: text encoder 8.4, denoiser forward 5.6, backward 34.2, clip + optimizer 1.9, between steps 1.1.
   The backward does 2x the forward's FLOPs and takes 6x its time.
2. d640 trace: 16 all-reduces per rank-step take 30.6 ms, 21.3 ms of it not under compute, at 14.3 GB/s (d1536: 55 GB/s).
3. On the rank that arrives last, the backward takes 16.5 ms (d640), so the all-reduce itself costs about 4 ms. The rest is
   ranks waiting in NCCL for the slowest rank.
4. Over 300 steps, step time has correlation 0.91 to 0.93 with the slowest rank's own compute, 0.42 to 0.46 with the mean.
5. Grouped batches give each rank its own T: mean 51, and the largest of 8 ranks averages 131. The encoder takes 10.9 ms
   mean per rank and 26 ms on the slowest rank of a step, at every model size. The denoiser's own spread is about 5 ms
   (3 x (7.84 - 6.12) at d640).

So the encoder inside the synchronous step sets the pace of every step, and small models pay most of it.

**Change.** Compute the outputs once (tools/encode_captions.py), and read them in the data workers
(GPICWebDataset(emb_dir=...), src/models/conditioner/cached_text.py). The training GPUs then run no text encoder.

**Math.** Real tokens carry the online encoder's values up to bf16 rounding: the same tokenizer, truncation, right
padding, batch size, kernel and autocast, in another batch mix. A causal mask keeps a real token from seeing padding, so
the batch mix does not change the values beyond rounding. Padded tokens are zeros instead of the encoder's values, and
the denoiser's masks keep them out of every attention (tools/test_text_cache.py check 5, bit-identical at fp32).
The null caption is stored once. Prediction keeps the online encoder.

**Cost.** Disk: 1.66 TiB (shard 0: 12639 captions, mean 44.0 tokens, 2.12 GiB; 180 KB per caption). The bytes have 2.81 + 7.97 bits
of entropy per 16 bits, so per-sample compression could save at most 33%, and raw bytes keep one code path. Compute: 28 s per
shard on a vis A100, so 801 shards take about 30 minutes on 8 H100s, once. Reads: each sample adds 180 KB to the data path.

## The image work comes after grouping (2026-11-03)
A worker that grouped by length read a whole window, 1024 samples, with the images decoded and cropped in it. That is about 6 s of CPU on one core (5.7 ms per image: crop 3.1, decode 1.6, tar and tensor the rest), and then the worker emits 32 batches. All workers start together, so all of them read together, once per 1024 steps at 32 workers per rank. The DataLoader delivers in order and each worker holds 4 batches ahead, so a rank can run on for 128 steps while its workers read. 256 workers on 96 CPUs take about 17 s to read, so at 32 ms per step (4.1 s of buffer) the rank waits about 13 s every 1024 steps: 45 ms per step in place of 32 ms. No smoke saw this, because every arm with 32 workers stopped before the second window (step 928). The 2-worker epoch arm showed it: a stall of 6 s every 64 steps.

Sorting needs only the text lengths. So the window now holds the compressed images, and each sample's decode and crop run when its batch is requested. A worker's CPU then follows the batches it emits: 160 ms per batch, and about 45 of the 96 CPUs at 32 ms per step. The first window of worker i holds (i + 1) / num_workers of a window, so the workers also read GPFS one after the other. Only the first window of each pass is smaller, which gives its batches a slightly larger T and does not change the math.

Same pixels: the decode is webdataset's own handler, and the tensors equal the old order's for the same RNG state (tmp/ladder_prof/same_pixels.py). Nothing after grouping may drop a sample, or a batch would lose its size, so an image that fails to decode repeats the sample before it, with a warning.

Measured on one rank with 16 workers, 7 CPUs and a target of 64 ms per step, which has the same buffer and the same load per CPU as 32 workers at 32 ms: images in the window, 82.3 ms per step, 6 s stall every 512 steps; first window staggered only, 92.4 ms, because in-order delivery then waits for each worker in turn. jobs/smoke_ladder_cached.sh measures both orders on 8 H100s.
