# Review: PixelGen ladder branch (f4dd854..fee1867), 2026-11-02

Scope: `git diff f4dd854 fee1867`, the unchanged code it calls (Lightning 2.6.6, torch 2.7.1, webdataset 1.0.2,
transformers 5.22.1, jsonargparse 4.41.0 in envs/pixgen), plus a light look at the owner's uncommitted working-tree
changes that appeared during the review (CompileWarmup callback, text encoder sdpa). Scripts: `tools/review/`.
After this review was written, the branch moved to ad258a8: 7eb89fe commits CompileWarmup and SDPA, and ad258a8
changes pad_multiple to 8 and recomputes the steps. I did not review ad258a8. S2 applies to 7eb89fe as committed.
N2 (budget spread from rounding to 4000) and the numbers in item 9 below refer to the fee1867 steps; re-check them
against the new docs/ladder_runs.json.

Verdict: no blocker found. The masked text path, CFG dropout, resume, LR and FLOP accounting are correct. Two
should-fix items (S1, S2) and several nits.

## Findings

### S1. `jobs/train_ladder.sh:46-48`: exit 0 is not tied to "max_steps reached"

- Scenario: Lightning handles SIGTERM itself. `_sigterm_notifier_fn` sets `received_sigterm`, and the loop raises
  `SIGTERMException`, a subclass of `SystemExit` with no code (`lightning/pytorch/utilities/exceptions.py:19`,
  `loops/training_epoch_loop.py:422`). `_call_and_handle_interrupt` re-raises it, so `python main.py` exits with
  **0** when it is stopped before the end. The script then runs `exit $rc`, so wq would mark the run finished and the
  ladder point would stop short without any error.
- Why it does not fire today: a TERM from timeout, wq or Slurm also reaches the parent bash. Bash has no TERM trap
  (env.sh only traps EXIT), so it dies at once with 143 or 124 before it reaches `rc=$?`. Evidence:
  `tools/review/sigterm_exit.sh` has a child that copies Lightning (catch TERM, finish the step, `raise SystemExit`).
  The host sees `rc=143` for an external TERM and `rc=124` when the time limit is reached, and "main.py exit" is never
  printed.
- Risk: the safety depends on bash dying before Python. Any wrapper that traps TERM and waits for the child, or a
  signal sent only to the Python ranks, turns a stopped run into a "finished" one.
- Fix (1 line): after `rc=$?`, add
  `[ $rc -eq 0 ] && ! ls $RUN/ms-step=$(printf %06d $STEPS)*.ckpt >/dev/null 2>&1 && { echo "exit 0 before max_steps"; rc=99; }`
  (the last ms checkpoint is written exactly at `STEPS`, since `STEPS % MS == 0`).

### S2 (PLAUSIBLE, uncommitted code). `src/callbacks/compile_warmup.py`: graphs compiled outside the DDP forward may skip DDPOptimizer

- Scenario: dynamo uses DDPOptimizer (splits the graph at DDP bucket boundaries so the gradient all-reduce overlaps
  the backward pass) only when a frame is compiled **inside** a DDP forward:
  `torch/_dynamo/convert_frame.py:1406-1427` checks `DistributedDataParallel._get_active_ddp_module()` at
  compile time. That check runs only on a cache miss. CompileWarmup calls `pl_module.denoiser` directly in
  `on_train_start`, outside DDP's `_inside_ddp_forward`. It compiles dynamic-T graphs at T = 64, 128 and 192. All
  real batches then hit those cached graphs, so I found no path that re-enters the DDPOptimizer branch. The result
  would be one monolithic compiled backward per step. All buckets reduce only after it ends, with no overlap.
- Cost: step time only, not correctness. It is larger for big sizes on 8 GPUs and changes the measured step times
  that set the job lengths.
- Evidence level: code reading only. My CPU/gloo reproduction (`tools/review/warmup_ddp_optimizer.py`) hung on
  vis1 and gave no result.
- Check (cheap, in the 8-GPU smoke): run with `TORCH_LOGS=ddp_graphs`. The DDPOptimizer split log appears only when
  it is active. Or compare d1280 step time with and without CompileWarmup.
- Fix if confirmed: run the warm-up through the DDP-wrapped module (e.g. a real `trainer.strategy.model` forward on
  dummy batches, with grads zeroed before step 1). Or set `torch._dynamo.config.optimize_ddp = False` for both paths,
  so the choice is explicit and the same everywhere.

### N1. `jobs/train_ladder.sh:39` (CB order): an ms checkpoint can be lost

The resume hook comes before the ms hook in the callbacks list. `MS` is a multiple of `LAST`, so both save at the
same step, and last.ckpt is written first. A kill between the two saves leaves last.ckpt at step k·MS with no
`ms-step=k·MS.ckpt`. The resumed run never saves that ms file again. `eval_ladder.sh` then fails loudly ("no ms
checkpoint"). The worst case is the final step. `STEPS` is a multiple of `LAST`, so a kill between the two saves at
`global_step == STEPS` leaves a full last.ckpt at max_steps and no final `ms-step=STEPS.ckpt`. The next launch
returns at once with exit 0 and never writes the file. The C0 point of that size is then lost. (With the S1 guard
the run would instead loop on 99 until `--max-runs`.) Fix: put the ms hook first in `CB`, in both
`train_ladder.sh` and `lite_ladder.yaml`. In the opposite race, the re-trained ms file becomes `ms-…-v1.ckpt`, and
`ls -t` in eval_ladder.sh already picks the newest one.

### N2. `tools/ladder_flops.py:89`: rounding to 4000 steps spreads the iso-FLOP budget by 1.6 points

`budget_flops/C0` is +0.4% for d640, +0.0% for d768, +0.8% for d896, +1.1% for d1024 and **+2.0% for d1280**. Fix:
round to a multiple of 32 (all the script needs), or fit with each run's actual `budget_flops`, not C0.

### N3. `jobs/eval_ladder.sh:21-24`: model dims are not checked against the checkpoint

`LightningModel._strict_loading = False`, so the state dict is loaded with `strict=False`. A wrong `nb` or `nt` gives
no size-mismatch error. Extra blocks keep their initialisation (zero adaLN, so they act as identity), and missing
blocks are silently dropped. The FD is then wrong with no error. Fix: count `ema_denoiser.blocks.*` and
`text_refine_blocks.*` in the ckpt before predict, or assert that `missing_keys` is empty.

### N4 (pre-existing, recipe property). The same RNG stream on every rank, restarted at every launch

`seed_everything: 0` gives every rank the same torch/CUDA seed (Lightning `reset_seed` in `strategies/ddp.py:200`).
So t, the noise and the CFG-dropout mask are identical across the 8 ranks at every step: 32 distinct (t, noise)
draws per 256-image step, not 256. Every relaunch also replays the stream from step 0. This is identical in the
reference run and in every ladder size, so the sweep stays internally comparable. Do not change it mid-sweep, but
record it, because new methods will be compared against this ladder.

### N5. `gpic_dataset.py:_tokenize`: no guard for a 0-token caption

An empty caption gives a mask row with no True, so that sample's text-refine attention sees only masked keys.
Measured: 0 of 50,215 captions in shards 0, 137, 400 and 655 have 0 tokens, and 1 has more than 320 tokens
(`tools/review/empty_caption.py`). On CPU the denoiser stays finite for an all-False row (torch 2.7 returns 0 for
fully masked rows). I did not check the CUDA mem-efficient kernel. Fix: map `[]` to `[eos]` in `_tokenize`.

### N6. Small items

- `configs_h100/lite_ladder_sample.yaml:224,247`: `pred_batch_size` appears twice (8, then 32; the last wins = 32).
  Inherited from main `lite_sample.yaml`.
- `jobs/eval_ladder.sh`: has `#FETCHABLE` but no `#SBATCH --comment=fetchable` (standing rule).
- `gpic_dataset.py:_group`: the chunk-order RNG is seeded with `hash((str, id, pid))`. String hashes are salted per
  process, so the order is not reproducible even with a fixed `GPIC_SHARD_SEED`. Harmless; use `zlib.crc32` if
  reproducibility is wanted.
- Long-caption positions 64-319 of `y_pos_embedding` get gradients only from batches with T > 64 (11% of batches;
  padded positions get none). This is inherent to option B and the same for all sizes.

## Checked and found correct

1. **Masked text end to end.**
   - Worker tokenization uses the same tokenizer class and call as the conditioner. Collate pads to
     ceil64(max len) ≤ 320 with the pad id, and the mask is True = real.
   - Lightning moves the nested metadata tensors to the GPU, so `_to_cuda` is a no-op.
   - The conditioner writes `y_mask` before `_impl_uncondition` reads T. The null embedding (one eos token) is cached.
   - CFG dropout replaces both the row and its mask, with the same Bernoulli draw.
   - `JiTTrainer` passes `y_mask`.
   - Denoiser: SDPA bool masks with True = attend. Text-refine uses [B,1,1,T]. Main attention uses [B,1,1,256+T]
     with image keys always valid. `y_pos` is sliced to T. With `y_mask=None` the call is unchanged.
   - Predict: strings are tokenized with pad multiple 64 and truncation at 320; `y_mask` is passed; with CFG ≠ 1 the
     mask becomes [null, cond], matching the sampler's [uncond, cond] order; CFG 1.0 runs the conditional branch only.
   - Predict uses the EMA weights (`eval_original_model: false`).
   - HF right padding with causal attention leaves no fully masked text query row, so masked keys carry no NaN values.
2. **Batching.** `group_window % batch_size == 0` is asserted, `group_batch` is forced to the DataLoader batch size,
   and nothing after `_group` drops a sample. So every per-GPU batch is exactly one sorted chunk, and B = 32 always
   reaches the compiled graphs.
3. **Padded T under the real stream order** (one worker reads a shard front to back, then shuffle(100), then windows
   of 1024). Mean T is 79.4, against 79.3 for the global shuffle in ladder_flops.py, on shards 0, 200, 400, 600 and
   800 (`tools/review/stream_T.py`). The histograms match. Caption types are interleaved in tar order (about 7,400
   runs per 12,600 samples), so windows are not type-skewed.
4. **Epoch ends with unequal rank streams.** With `limit_val_batches 0` and `check_val_every_n_epoch None`, no
   validation runs. ModelCheckpoint `on_train_epoch_end` does nothing (`every_n_epochs` = 0; the last-save branch is
   skipped by `_last_global_step_saved == global_step`). No metric is logged `on_epoch`. The per-caption-type
   `sync_dist` logs run on `should_update_logs`, which depends only on global_step and is the same on all ranks.
   `setup_data` (with its barrier) runs once. Each rank takes one step per batch, so the DDP collectives stay matched.
   The only per-rank difference is `current_epoch`, and nothing uses it.
5. **EpochShuffledShards.** It is pickled into each worker, and persistent workers call `__iter__` once per epoch.
   So every rank and worker copy uses the permutation `seed*100003 + pass`, and split_by_node/worker covers each
   shard once per epoch (test 6). `GPIC_SHARD_SEED` is exported before Lightning spawns the ranks, and is set inside
   the script, so `env -i` in fetch_pending does not lose it.
6. **The callbacks JSON on the command line** replaces the YAML list. It does not append (`tools/review/cli_callbacks.sh`,
   run through LightningCLI `--print_config` on vis1). Output: two CheckpointHooks with `every_n_train_steps`
   16125 / 129000, `save_top_k` 0 / -1, `save_last` True / False. In the same run, a later
   `--trainer.logger.init_args.id` wins (smoke ids do not collide with production ids), and the denoiser dims are
   640/10/9/2 with txt_max_length 320.
7. **Resume.**
   - last.ckpt keeps the full state (CheckpointHook keys on the file name, through a class attribute), and the ms
     files keep the state_dict (model + EMA) only.
   - The restore happens after `configure_model`'s EMA copy, so the EMA comes from the checkpoint.
   - The LambdaLR `last_epoch` is restored.
   - A resume at `global_step == max_steps` returns at once with 0, so re-running a finished run is safe.
   - The Lightning checkpoint save is atomic (fsspec transaction), so a kill during a save leaves the previous
     last.ckpt intact.
8. **LR.** `WarmupFlatCooldown(4000, 4000, 1e9)` is linear 0→1 over 4000 steps, then flat for any max_steps in the
   sweep (the cooldown would start at step 999,996,000).
9. **FLOPs.** C0 = 40000 × 256 × 3 × fwd(1536, 16, 4, T = 300) = 1.4258e19. The per-size steps follow from the
   measured E[T] and E[T²]. TPP = steps × 256 × 256 / N. All max_steps are multiples of 32, and MS = STEPS/4 is a
   multiple of LAST = STEPS/32, so the last ms checkpoint lands exactly at max_steps.
10. **Eval checkpoint selection.** `ms-*$(printf %06d step)*.ckpt` matches the `ms-step=XXXXXX.ckpt` names, and
    `-v1` re-saves sort newest-first. The ms files have no "callbacks" key, which predict restore tolerates.
11. **Job scripts.**
    - The health gate exits 98.
    - Exit codes under host signals are 143/124 (S1 evidence).
    - `PYTHONPATH` is set to the frozen worktree.
    - The divisibility check is present.
    - eval runs with `set -e`, so a failed predict stops the job before fast_eval.
12. **Data edge cases.** In 4 shards there are no empty captions and 1 caption over 320 tokens. On CPU, an all-False
    mask row stays finite in forward and backward.
