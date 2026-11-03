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

**Cost.** Disk: see the job's totals. Reads: each sample adds its bytes to the data path.
