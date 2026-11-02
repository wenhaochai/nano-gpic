# Implementation spec: length-grouped, masked text for the PixelGen ladder (option B)

Worktree: `/scratch/gpfs/GROUP/USER/project/gpic-wt/ladder` (branch `ladder`, code in `baselines/PixelGen`).
This branch is the official GPIC PixelGen (JiT-T2I) code plus exact-math speedups C1-C8, the fixed CheckpointHook and
an optional shard shuffle (`GPIC_SHARD_SEED` in `src/data/dataset/gpic_dataset.py`). Read `docs/ladder_plan.md` first
(owner-approved plan; section 2 option B is what you build).

Rules: commit on branch `ladder` (several focused commits, each with tests passing), push to origin. Do not submit
GPU jobs, do not touch other branches or worktrees. The login node only edits and runs git; run Python on vis2
(`ssh vis2 'cd <worktree> && source /scratch/gpfs/GROUP/USER/envs/pixgen/bin/activate && export
PYTHONPATH=$PWD/baselines/PixelGen HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 && nice -n
19 python ...'`; two A40s there, CUDA_VISIBLE_DEVICES=0 or 0,1, or empty for CPU). Never `source jobs/env.sh` on a vis
node. Run every long step in the foreground. The A40 cannot run the text encoder's FlexAttention kernel (shared memory),
so tests that run the text encoder use attn_implementation="sdpa" (add that option, below). Write your report to
`docs/ladder_impl_report.md` (what changed, file:line, each test and its numbers, anything not done).

Everything below is OFF by default: with the new options unset, every code path must stay bit-identical to the
current branch (test it).

## 1. Data: tokenize in the workers, group by length (`src/data/dataset/gpic_dataset.py`)

1. `GPICWebDataset(..., group_text_len: bool = False, group_window: int = 1024, group_batch: int = 32,
   text_max_length: int = 320, tokenizer_path: str = "Qwen/Qwen3-1.7B")`.
2. When `group_text_len`: after the existing caption/transform stages and the sample shuffle, tokenize each caption in
   the worker with the same Qwen tokenizer the text encoder uses (load once per worker process; same call as
   `src/models/conditioner/qwen3_text_encoder.py` uses, without padding; truncation only at `text_max_length`, count and
   print how many were cut). Store the ids in the sample.
3. Then a grouping stage: buffer `group_window` samples, sort them by token length, cut into chunks of `group_batch`
   consecutive samples, shuffle the chunk order (seeded per worker and per `GPIC_SHARD_SEED` if set), and emit chunk by
   chunk; flush the remainder at the end of the stream the same way.
4. Each per-GPU batch (train_batch_size 32 per GPU in the configs) must be exactly one chunk. Check how
   `build_dataloader` batches (DataLoader batch_size per worker, or a pipeline `.batched`) and make the alignment hold;
   assert it in a test.
5. Collate: images stacked as now, captions as a list, metadata as now plus `input_ids` [B, T] and `text_mask` [B, T]
   (bool, True = real token), T = the batch's longest caption rounded up to a multiple of `pad_multiple` (default 64,
   config option), padded with the tokenizer's pad id. Pinned like the rest of the batch (C1 loader).

## 2. Text encoder (`src/models/conditioner/qwen3_text_encoder.py`)

1. New init options: `text_mode: str = "official"` ("official" = current behaviour: pad/truncate to max_length=300;
   "masked" = the new path) and `attn_implementation: str = "flex_attention"` (tests use "sdpa").
2. "masked": when the batch metadata carries `input_ids`/`text_mask`, use them (pinned, non_blocking copy, as C7);
   for string inputs (prediction), tokenize with padding to the longest caption rounded up to a multiple of 64, no
   truncation below 320. Return the last hidden state [B, T, 2048] and put the mask in metadata (`y_mask`).
3. Null caption for CFG dropout in "masked" mode: the single token `tokenizer.eos_token_id` (encode it once, cache the
   [1, 1, 2048] embedding). Never a caption with zero valid tokens.
4. torch.compile: T takes a handful of values (multiples of 64 up to 320). Make sure this does not recompile on every
   new T beyond those few, and raise `torch._dynamo.config.cache_size_limit` if needed; report the number of graphs.

## 3. CFG dropout (`src/diffusion/base/training.py` BaseTrainer.preproprocess)

In "masked" mode (metadata has `y_mask`): for dropped samples (same Bernoulli(null_condition_p) as now), set the
condition row to the null embedding at position 0 and zeros after, and its mask row to [True, False, ..., False].
Official mode unchanged.

## 4. Denoiser (`src/models/transformer/JiT_T2I.py`)

1. `forward(x, t, y, y_mask=None, return_layer=None)`; `y_mask` [B, T] bool or None.
2. `y_pos_embedding` keeps its shape (txt_max_length; the ladder config sets 320); slice `[:, :T]`.
3. TextRefineAttention: key padding mask from `y_mask` (query rows of padding still get outputs; they are masked as
   keys everywhere). Main Attention: keys are [256 image tokens, T text tokens]; image keys always valid, text keys
   per `y_mask`. Use SDPA with a boolean attn_mask broadcast as [B, 1, 1, keys]. `y_mask=None` must keep the current
   unmasked call (no mask argument at all), so the official path is unchanged.
4. JiTTrainer (`src/diffusion/flow_matching/training_JiT.py`) and the sampler call path (`predict_step` in
   `src/lightning_model_gpic.py`, `EulerSamplerJiT`): pass `y_mask` through to every denoiser call (e.g.
   functools.partial on the net in predict_step). Also the EMA denoiser.

## 5. Configs and jobs

1. `configs_h100/lite_ladder.yaml`: from main `configs_h100/lite_ref_e1.yaml` (in git: `git show
   origin/main:baselines/PixelGen/configs_h100/lite_ref_e1.yaml`) plus: strategy DDPStrategy(broadcast_buffers
   false, gradient_as_bucket_view true, ddp_comm_hook src.utils.ddp_hooks.allreduce_avg_hook); AdamW fused true;
   enable_progress_bar false; limit_val_batches 0, num_sanity_val_steps 0, val_check_interval 1e9; model.sample_every
   0; dataset group_text_len true, pad_multiple 64; conditioner text_mode masked, max_length 320; denoiser
   txt_max_length 320; logger WandbLogger project gpic-speed; callbacks: CheckpointHook last every 2500 steps
   (save_last true, save_top_k 0) and CheckpointHook ms-{step} every N steps (save_top_k -1, N set per run). The LR
   scheduler stays WarmupFlatCooldown(4000, 4000, 400000) exactly as the reference.
2. `configs_h100/lite_ladder_sample.yaml`: from main `lite_sample.yaml` with the masked text mode and
   txt_max_length 320 (model sizes come from the command line).
3. `jobs/train_ladder.sh <size> <d> <nb> <nt> <max_steps>`: 8 GPUs, `-c 96`, `--comment=fetchable-partial`,
   `#FETCHABLE`, `source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh`, GPU health gate as in
   `jobs/bench8_c8.sh` (exit 98), `export GPIC_SHARD_SEED=$(date +%s)`, run dir `runs/gpic-speed/exp_ladder_<size>`
   (main.py takes `--tags.exp`, not `--exp`), resume from the newest `last*.ckpt` by global_step, num_groups = d/64,
   ms checkpoints every max_steps/4, W&B name/id ladder_<size>. Exit with main.py's code.
4. `jobs/eval_ladder.sh <size> <d> <nb> <nt> <step>`: like `project/gpic/jobs/sample_eval.sh` (8 GPUs, EMA weights,
   predict with save_compressed, then `tools/fast_eval.py`), with the size arguments and lite_ladder_sample.yaml.

## 6. FLOPs and steps (`tools/ladder_flops.py`)

1. The analytic per-image training FLOPs of the denoiser as a function of (d, nb, nt, T) (the formula in
   `docs/ladder_plan.md`: main block 32·L·d² + 4·T·d² + 4·L·(L+T)·d, text block 32·T·d² + 4·T²·d, embeddings; ×3 for
   training; it gives 1392 GFLOP for d1536 16+4 at T=300) and the parameter count (check against the real module).
2. Measure the padded T distribution produced by the real pipeline (group_text_len, pad_multiple 64) over at least
   2000 batches from several shards (CPU, no images needed if you can skip decoding; otherwise decode).
3. Print, for d640 9+2, d768 10+2, d896 11+2, d1024 12+3, d1280 14+3: GFLOP/img at the measured mean T, max_steps for
   C0 = 1.426e19 FLOPs at batch 256 (round to a multiple of 4000), TPP, Lite epochs. Save as `docs/ladder_runs.json`.

## 7. Tests (`tools/test_ladder.py`, all must pass; report numbers)

1. Official mode unchanged: with every new option at its default, denoiser and text-encoder outputs and one
   training-step loss equal the current branch bit for bit (fp32, small model, CPU or A40).
2. Padding invariance (fp32): the denoiser output for a sample does not depend on how many padded text positions follow
   it (T = 64 vs T = 128 vs T = 320, same caption): max |diff| at fp32 rounding level. Same for the text encoder's
   valid positions (sdpa).
3. CFG dropout: dropped rows carry exactly the null embedding at position 0 and a mask with one True.
4. Grouping: every per-GPU batch comes from one chunk (lengths sorted within the window); report the mean padded T.
5. A 2-rank Lightning smoke on the two A40s with lite_ladder.yaml, a tiny denoiser (num_blocks 2, num_text_blocks 1,
   d 256 or the smallest that works), text encoder sdpa, 30 steps, max_steps small: finishes, loss finite, a resume
   from last.ckpt continues the step count, and the number of compiled graphs stays small.
