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

Measured on one rank with 16 workers, 7 CPUs and a target of 64 ms per step, which has the same buffer and the same load per CPU as 32 workers at 32 ms: images in the window, 82.3 ms per step, 6 s stall every 512 steps; first window staggered only, 92.4 ms, because in-order delivery then waits for each worker in turn; images after grouping, 69.5 ms, with no stall over 0.3 s after the start. Out-of-order delivery adds nothing to that (69.0 ms), so delivery stays in order. jobs/smoke_ladder_cached.sh measures both orders on 8 H100s.

## The C++ launcher (2026-11-03)
The C++ kernel launcher (TORCHINDUCTOR_CPP_WRAPPER=1) takes d640 from 39.3 to 32.6 ms. It computes the same math up to bf16 rounding, not bit for bit (tools/test_cpp_launcher.py).
Correction: this section first said that d640's step is bound by the host, "37.6 of 38.6 ms on every rank". That number is the wall time between host markers, which adds up to the step time by construction, so it does not show what binds the step. See "What is left of the step".

## Host settings for production (2026-11-03, w974)
Each arm changes one thing on the cached path with the C++ launcher, over the same steps, without checkpoint saves. d640, mean / median ms per step: control 35.9 / 32.4; gc.freeze after the warm-up 35.7 / 32.3, with no 0.5 s stalls; log_dict only on written steps 36.9 / 32.2 (3.1 s of stalls); tcmalloc 35.9 / 31.1; rank and workers on the GPU's socket 34.4 / 30.3; DDP buckets of 100 MiB 32.9 / 29.6, of 1000 MiB 32.3 / 28.6; all but the buckets 31.5 / 28.3; all with 100 MiB 30.1 / 27.0. Lightning's per-step SIGTERM broadcast off: median 32.5, so it stays. d1280: control 92.7 / 91.6; 1000 MiB 88.4 / 88.9; all but the buckets 91.5 / 92.2. Production: jobs/train_ladder.sh and src/callbacks/host_tuning.py, with buckets of 1000 MiB. The combination of all of them with 1000 MiB has no arm; the ladder runs themselves give it. Not done: CUDA graphs in the denoiser (the arm did not finish 1200 steps in 11 min; 40 T values times the graphs), and a Python broadcast patch. At 30 ms per step the data workers use about 70 of the 96 CPUs, so the data path binds near 25 ms.

## What is left of the step (2026-11-03, w1048, node-k16g2, code 89dbf49, review docs/review_kernels_2026-11-03.md)
Production settings of jobs/train_ladder.sh, steps from 100 on, without checkpoint saves. Mean / median / p90 / p99 ms per step:
- d640 production 29.8 / 26.8 / 41.9 / 73.5 over 5894 steps. Mean minus median stays 2.9-3.4 ms in every sixth of the run, so the slow steps are not one-time costs (an earlier guess from 1300 steps).
- d640 with no host-side data work (64 real pinned batches in memory; no workers, pin thread or shared memory; the copy to the GPU stays) 25.1 / 25.3 / 27.8 / 55.3. This is the bound for data-path work: 4.7 ms, 16%. The median within one batch of the 64-step cycle varies by 0.06 ms and across batches from 22.5 to 30.3 ms, so without data workers a step's time is set by its batches.
- d640 with the main and autograd threads alone on one CPU per rank, 44 CPUs per socket left for workers: 31.2 / 25.0 / 32.9 / 164.6, with 15 steps over 0.3 s, 8.1 s in sum. CPU wait of the two threads falls from 8.5% and 7.7% of the wall clock to 0.4% each, and p90 falls by 9 ms, so CPU contention causes the p75-p90 steps. The stalls are one rank late for its batch; the workers' summed wait for a CPU rose from 4.3 to 37.1 CPU-equivalents, cause not found. Not for production as it is. With no workers, pinning changes nothing (25.3 / 25.0).
- d640 with aggressive fusion 29.8 / 26.3 / 42.5 / 78.8, with one graph per direction 29.8 / 26.8 / 41.7 / 80.5. Neither reduces the kernels: per rank 218 forward and 354-359 backward Triton kernels with fusion, 222 and 358-363 with one graph, against 218 and 358 in production.
- d640 with GEMM autotuning: no steps. The cold compile took 1016 s for T=64 and 408 s for T=128, and the arm's limit of 1500 s came in T=320. Its graphs have 25-28 forward and 57-70 backward templates. Not measured.
- d1280 production 88.1 / 88.8 / 96.0 / 156.1 over 1098 steps.
The trace (3 steps per rank, first step dropped):
- d640: 1070 kernels per rank-step, sum of kernel time 19.3 ms: element-wise 8.2 ms (765 kernels, 10.7 us each), GEMM 5.4 ms (219), attention 2.9 ms (22), all-reduce 2.2 ms (1), other 0.6 ms. GPU busy 16-21 ms of a 21-23 ms step at T of 16-64; idle in gaps under 10 us about 2.3 ms (1000 gaps). Only 2-11% of kernels start within 5 us of their launch, and the median kernel starts about 2 ms after it, so in these steps the GPU binds, not the host launching. Three steps at small T are a small sample.
- d1280: 88-89 ms busy of 92 ms in the clean step, 1760 kernels; 3 all-reduces, the first 5.6 ms after the last rank arrives.
- All-reduce at d640: 438 MB in 1.75 ms after the last rank arrives, about 250 GB/s; the rest of its 2.2-2.7 ms kernel time is ranks arriving 1.2-2.0 ms apart. The kernel name RING_LL misleads: w874's TUNING logs (2026-11-01) show Ring/Simple, and NVLS for all-reduce gave no gain.
Main thread by library (perf, 15 s, rank 3): python 32.7%, libcuda 19.3%, libtorch_cpu 9.9%, libtorch_python 8.7%, libc 8.5%, libtorch_cuda 4.9%, libc10_cuda 4.8%, tcmalloc 4.1%. Autograd thread: libcuda 27.2%, libc 17.9% (mutexes 9.2%), libtorch_cpu 17.0%, libc10_cuda 7.9%, cuBLAS 6.5%.
One step in 50 costs about 30 ms at d640 and 55 ms at d1280: the SIGTERM broadcast of src/utils/patch_bugs.py, a host sync. That is 0.57 and 1.31 ms per step on average.
Levers left, none tested: (1) the data path, up to 4.7 ms at d640: lower priority for workers instead of fewer CPUs, fewer workers, an exact faster crop; (2) one DDP bucket at d1024 and d1280: three buckets at d1280 take about 15 ms (estimate: the first takes 5.6 ms), and the flat rate of 272 GB/s (2026-11-01, tools/prof/nccl_bench.py) gives about 10 ms for 2.66 GB; (3) the broadcast every 200-500 steps.
