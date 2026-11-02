# Adversarial review: Muon ablation code (branch muon @ 595f3cb)

Reviewer: independent session, 2026-11-02. Scope per runs/gpic-speed/page/muon_review_task.md.
Scripts used for evidence: `/scratch/gpfs/GROUP/USER/tmp/review_muon/*.py` (vis2; CPU or one A40).

Summary: the optimizer math, routing, part split, distributed plan, AdamW equivalence and FixedValLoss are correct.
The job path is not: no job can start (B3), and no arm can resume (B1: last.ckpt has no optimizer state; B2:
MuonAdamW's step tensors land on the CPU; B4: resumes keep reading the stale last.ckpt). The smoke as written
would miss B1 and B4 and exit 0 anyway. Separately, bf16 Newton-Schulz on the rank-deficient adaLN gradients gives
updates ~10x below the 0.2 lr target (S1), which changes what the ablation measures.

## Blockers

### B1. last.ckpt has no optimizer state, no LR-scheduler state and no loop state, so every resume of a 20k arm fails
- Where: `baselines/PixelGen/src/callbacks/model_checkpoint.py:59-72` together with `configs_h100/lite_ablation.yaml:52-65` (two CheckpointHook instances).
- Mechanism: Lightning calls `on_save_checkpoint` of **every** callback for **every** checkpoint
  (`lightning/pytorch/trainer/call.py:253-255`, `_call_callbacks_on_save_checkpoint` loops over `trainer.callbacks`;
  called from `checkpoint_connector.py:509` for any save without weights_only). `_saving_last_ckpt` is per instance.
  When hook 1 (`resume-*`, save_last) writes `last.ckpt`, its own flag is True, but hook 2 (`ms-*`) also runs
  `on_save_checkpoint` with its flag False, and pops `optimizer_states`, `lr_schedulers` and `loops` from the same dict.
- Failure: the first resume (`train_arm.sh` passes `--ckpt_path $RUN/last.ckpt`) hits
  `checkpoint_connector.py:374-378`: `KeyError: Trying to restore optimizer state but checkpoint contains only the model`.
  `train_arm.sh` is `fetchable-partial` with a 2 h `--time`, so a 20k-step arm (~0.3 s/step = ~1.7 h plus compile and
  validation) that is cut short, or run in a host for less than its full time, can never finish: it crashes on every restart.
  If the KeyError were bypassed, the loss of `loops` would restart global_step at 0.
- Why the smoke does not catch it: `lite_ablation_smoke.yaml:4-14` replaces the whole `callbacks` list (jsonargparse
  replaces lists) with **one** CheckpointHook, so the smoke resume test (smoke_muon 200 -> 300) runs with a single hook
  and passes. The 20k config is the only one with two hooks. The reference run `lite_ref_e1.yaml:48-61` has the same
  two hooks but never resumed (one 3 h 45 min job, `logs/lite_ref_e1.yaml`), so the bug was never exercised.
- Evidence (confirmed): `/scratch/gpfs/GROUP/USER/tmp/review_muon/ckpt_check.py` on vis2 (CPU,
  Lightning BoringModel, the repo's CheckpointHook twice as in lite_ablation.yaml, scaled to resume every 4 steps and ms
  every 8; 12 steps, then resume to 20):
  ```
  segment 1 files: ['last.ckpt', 'ms-step=000008.ckpt']
  last.ckpt keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict'] global_step 12
  resume: FAIL KeyError 'Trying to restore optimizer state but checkpoint contains only the model. ...'
  ```
- Fix direction (not applied): decide by filepath, not by instance flag (e.g. in `on_save_checkpoint` keep the state
  when `trainer`'s current save target is a `last*.ckpt`, or give the ms hook `save_weights_only`-style stripping only
  inside its own `_save_checkpoint`); and make the smoke overlay keep both hooks.

### B2. MuonAdamW crashes on the first step after any resume: the AdamW `step` tensors come back on the CPU
- Where: `baselines/PixelGen/src/optim/muon.py:84,92,177-178,187-188`.
- Mechanism: with fused AdamW the per-param `step` must be a CUDA tensor. The param groups of MuonAdamW have no
  `fused` (or `capturable`) key, so `torch.optim.Optimizer._process_value_according_to_param_policy`
  (`torch/optim/optimizer.py:740-752`, torch 2.7.1) leaves `step` where `torch.load` put it. Lightning loads with
  `map_location=lambda storage, loc: storage` (`lightning/fabric/plugins/io/torch_io.py:64`), i.e. CPU, and
  `_optimizer_to_device` deliberately skips the key `step` (`lightning/fabric/utilities/optimizer.py:38-41`).
  torch.optim.AdamW(fused=True) avoids this because its groups carry `fused=True`.
- Evidence: `/scratch/gpfs/GROUP/USER/tmp/review_muon/resume_check.py` on vis2, one A40, replays
  Lightning's path (torch.save -> load with CPU storage -> load_state_dict -> _optimizer_to_device -> step):
  ```
  before save, adam step devices: {'cuda:0'}
  after load, adam step devices: {'cpu'}
  step after resume: FAIL RuntimeError Expected all tensors to be on the same device, but found at least two devices,
    cuda:0 and cpu! (when checking argument for argument state_steps in method wrapper_CUDA___fused_adamw_)
  control torch AdamW step devices: {'cuda:0'}
  ```
- Failure: every Muon arm dies on its first step after a resume (and the smoke step `smoke_muon_resume` will fail). The
  AdamW arm is not affected. `tools/test_muon.py:115-118` misses it because it round-trips the state dict in memory
  with `copy.deepcopy`, which keeps `step` on CUDA.
- Fix direction: put `fused=self.fused_adamw` into each AdamW group (then torch moves `step` to the param device), or
  move `step` to `p.device` in `_adamw_step`; test with a save/load through CPU storage.

### B3. `train_arm.sh` passes `--exp`, which main.py does not accept: every arm and the whole smoke exit at argument parsing
- Where: `jobs/train_arm.sh:26` (`--exp abl_$ARM`). main.py registers the tag as `tags.exp`
  (`baselines/PixelGen/main.py:36-39`, `parser.add_class_arguments(TagsClass, nested_key="tags")`).
- Evidence (vis2, CPU, `--print_config` stops before any model is built):
  ```
  python main.py fit --config configs_h100/lite_ablation.yaml --config configs_h100/lite_ablation_muon.yaml \
      --exp abl_x --trainer.logger.init_args.id abl-x --print_config
  -> error: Unrecognized arguments: --exp abl_x
  same command with --tags.exp abl_x -> parses; tags: {exp: abl_x}
  ```
- Failure: all four smoke steps and all five arms fail in seconds. `smoke_ablation.sh` ignores each step's rc and ends
  with `log DONE`, so the smoke job exits 0 even when every step failed (see S-list). Note also: with `--tags.exp abl_X` the run directory is
  `default_root_dir/` + `"exp_" + "abl_X"` (main.py:54-57) = `runs/gpic-speed/exp_abl_X`, which matches `$RUN` in
  train_arm.sh, so the fix is just the flag name.

### B4. After the first resume, Lightning writes `last-v1.ckpt`, and `train_arm.sh` keeps resuming from the stale `last.ckpt`
- Where: `jobs/train_arm.sh:23` (always `--ckpt_path $RUN/last.ckpt`) with `src/callbacks/model_checkpoint.py:65`
  (`checkpoint.pop("callbacks")`). ModelCheckpoint's version counter (`model_checkpoint.py:908-912` in Lightning)
  picks `last-vN.ckpt` whenever `last.ckpt` exists and is not its own `last_model_path`. A resumed run gets
  `last_model_path` from the callback state, and the hook deletes that state, so the resumed run never overwrites last.ckpt.
- Evidence (confirmed): `/scratch/gpfs/GROUP/USER/tmp/review_muon/ckpt_check2.py` on vis2 (CPU, one
  CheckpointHook as in the smoke overlay; fit to 8, resume to 16, resume to 24, all from last.ckpt as train_arm.sh does):
  ```
  after fit to 8 (from None): {'last.ckpt': 8}
  after fit to 16 (from last.ckpt): {'last-v1.ckpt': 16, 'last.ckpt': 8}
  after fit to 24 (from last.ckpt): {'last-v1.ckpt': 16, 'last-v2.ckpt': 24, 'last.ckpt': 8}
  ```
- Failure: with B1 fixed, the 2nd segment of an arm restarts from the 1st checkpoint ever written, the 3rd segment
  also, and so on: a chain of partial segments never moves past the step of the first last.ckpt (work lost, and the
  `val_loss.jsonl` gets duplicated steps). The smoke covers only one resume and reads no file name, so it passes.
  The resumed segment also opens `last.ckpt` at step 200 but then writes `last-v1.ckpt` at 300, so any later check
  of `last.ckpt` in the smoke shows step 200.
- Fix direction: keep the ModelCheckpoint state (do not pop `callbacks`, or pop only for non-last files), or
  `enable_version_counter=False` on the resume hook, and let train_arm.sh pick the newest `last*.ckpt` by global_step.

## Should-fix

### S1. bf16 Newton-Schulz is far from msign on low-rank inputs; the adaLN matrices get updates 5-20x below the intended RMS
- Where: `src/optim/muon.py:50-64` (bf16 default), `:7-10` (claim: update RMS = 0.2 lr). The adaLN gradient is
  `sum_b outer(dmod_b, c_b)` with `c = silu(t_embedder(t))`, which depends on t only.
- Evidence: `tmp/review_muon/numerics_check.py` and `adaln_rank.py`, vis2, one A40:
  ```
  msign (9216, 1536) rank=None: bf16 vs fp32 rel 0.017, cos(bf16, UV^T) 0.984, update RMS/lr 0.187
  msign (9216, 1536) rank=256 : bf16 vs fp32 rel 0.131, cos(bf16, UV^T) 0.400, update RMS/lr 0.078
  (same pattern at 4608x1536, 12288x1536, 1536x6144, 1536x1536)
  c(t) at init: singular values / max  >1e-1: 1, >1e-2: 2, >1e-3: 3 of 1536
  msign(adaLN grad) singular values >0.5: 3; update RMS/lr: bf16 0.012, fp32 0.009
  ```
  Full-rank inputs are fine (cos 0.98, RMS 0.19). For a rank-r input, 5 quintic steps lift only r singular values to
  ~1, so the update RMS is 0.2 lr sqrt(r / min(d)) (rank 256 -> 0.078; rank 3 -> ~0.01), and in bf16 the
  normalized small singular values are noise (cos to UV^T 0.40 at rank 256).
- Consequence: (a) the claim in the docstring that Muon's update RMS matches AdamW's is false for adaLN
  (adaLN_modulation, 9216x1536 x 20 blocks = 283M params). The other 6 kinds take token activations as input
  (256 images x 256 tokens, or text tokens), so their gradients are expected to be near full rank; I did not measure
  them on real gradients. The
  matrix whose function is the time conditioning effectively trains ~10x slower than under AdamW (AdamW normalizes
  per entry, so its RMS stays ~0.2 lr regardless of rank). An ablation result "Muon worse/better" will then partly
  reflect this. (b) Only adaLN's input is a curve in t: every token of an image shares one c(t), so its gradient
  has rank <= 32 per rank and <= 256 after the DDP average, and far less in effect (above).
- Suggestion: report this; either route adaLN to AdamW (as Moonlight / Keller Jordan route embeddings and heads),
  or accept and state it. A run-time log of the per-group update RMS would show it directly.
  Not verified: the rank of c(t) in a trained model. `tmp/review_muon/adaln_rank_trained.py` (loads the
  t_embedder from `exp_lite_ref_e1/ms-step=040000.ckpt`, 9.0 GB) did not finish in 20 min on vis2 (load
  average ~30) and was stopped. The init numbers above are from the same init as the arms (normal std 0.02), so S1
  holds at least for the early phase, which includes the 4k-step warmup. Run that script (or log the update RMS of
  each group in the smoke) before concluding for later training.

### S2. `smoke_ablation.sh` always exits 0 and deletes run directories without any guard against wrong `$WC`
- Where: `jobs/smoke_ablation.sh:79-83`: `rc=$?` is printed but never propagated; `log DONE` exits 0. `train_arm.sh`'s
  `exit $rc` is lost. A wq/fetch ledger records success even when all four steps failed (they would today, B3).
- `:77` deletes `$WC/runs/gpic-speed/exp_abl_smoke_*` with `rm -rf`. The `case` guard limits it to the smoke dirs, good;
  but `$WC` comes from env.sh, so if env.sh fails to source (no `set -e`, `set -u`), `$d` is `/runs/gpic-speed/...`,
  harmless but silent. Minor; the rc propagation is the real issue.

### S3. LR schedule: 20k steps of a 400k schedule = 4k warmup then flat, no cooldown; Muon group is scaled, as intended
- `WarmupFlatCooldown` is a `LambdaLR` with one lambda, applied to every param group including the Muon group
  (`src/utils/lr_scheduler.py`; LambdaLR multiplies each group's `initial_lr`). So the Muon group gets the same warmup:
  correct per the spec. `muon_lr` becomes the group's `initial_lr` at scheduler creation, fine.
- But note for the claim: `max_steps: 400000` keeps the official schedule, so the arms end at peak LR (no cooldown). FD at
  10k/20k is therefore taken before any decay; compare arms only at matched steps (the plan does). Also, AdamW's first 4k
  steps of warmup are 20% of the run: an optimizer comparison here is dominated by warmup + flat phase. Not a bug;
  state it.

### S4. Each resume restarts the data stream at the first shards: a resumed arm sees different data than an unbroken arm
- Where: `src/data/dataset/gpic_dataset.py` `create_dataset`: `SimpleShardList` in fixed order, `split_by_node`,
  `split_by_worker`, a 100-sample shuffle buffer seeded from pid+time (webdataset `filters.py` `_shuffle`), and no
  data state in the checkpoint. A segment that starts at step 10000 replays the shards read in steps 0-10000.
- Failure: with B1/B2/B4 fixed, arms that were cut by the host or the 2 h `--time` see a repeated first part of
  GPIC Lite, and the others see fresh data: the comparison between arms gets a confound that depends on scheduling,
  not on the optimizer. 20k x 256 = 5.1M samples is about half of one Lite epoch (the 40k reference run = 1 epoch),
  so an unbroken arm never repeats data, and a broken one does.
- Suggestion: run each arm in one segment (give `--time` a margin over the measured s/step, and do not mark it
  partial; wq's rule asks for `--partial` above 3 h, so a > 3 h arm needs the owner's OK or the shard offset), or skip the consumed shards on resume (offset by `global_step`). At least
  log the number of segments of each arm.

### S5. `train_arm.sh --time=02:00:00` is tight for 20k steps, so arms will need resumes, which hit B1/B2/B4/S4
- The reference run took 3 h 45 min for 40k steps incl. sampling and 1k-step checkpoints (`logs/lite_ref_e1.yaml`), i.e.
  ~0.34 s/step; 20k steps = ~1.9 h before compile (~several min), 20 FixedValLoss evaluations (2 nets x 8 levels x
  64 images per rank) and Muon's Newton-Schulz plus one all-gather per step. The ablation code has the C1-C7 speedups,
  so its step is faster than the reference's; I did not measure Muon's step time. Whether 2 h is enough is unproven
  until the smoke reports s/step for each optimizer. Size `--time` from that number.

### N1. `train/lr` logs only `param_groups[0]` and FixedValLoss's first evaluation is at step 1 only if step 1 is hit
- `lightning_model_gpic.py:234` logs group 0's lr (AdamW). Fine as both share one lambda; with a different `muon_lr`
  (planned arm `muon_lr2e-4` with AdamW 1e-4?) the Muon LR is never logged. Check which arms vary which LR:
  `abl_muon_lr2e-4` vs `abl_adamw_lr2e-4` need `--model.optimizer.init_args.lr`/`muon_lr` overrides that
  `train_arm.sh` passes through `"$@"`; the logs/*.yaml do not record the override. Record it.
- `val_loss.py:113`: `step == 1 or step % every == 0`. On a resume at step 2500, the 3000 evaluation happens, fine;
  the step-1 baseline is not repeated, fine.

### N2. `eval_arm.sh` picks `ls ms-*$STEP*.ckpt | head -1`; a resumed arm that re-saves the same step writes `ms-step=010000-v1.ckpt`
- With S4/B4 a re-run of steps 10000+ can produce a second ms file; `head -1` picks the first by name. Minor once the
  resume path is fixed. `eval_arm.sh` has `#FETCHABLE` but no `--comment=fetchable` line (train_arm has its comment).

### N3. `train_arm.sh` runs whatever HEAD of `gpic-wt/abl_run` is, with no check against the reviewed commit
- `abl_run` is now at 595f3cb, i.e. it holds B1-B4. After fixes, update abl_run and have train_arm.sh refuse a dirty
  tree or a SHA other than an expected one (it only records `commit.txt`, overwritten per segment).

### N4. After `load_state_dict`, the param-group settings from the checkpoint override the config
- torch `Optimizer.load_state_dict` restores `split`, `momentum`, `ns_steps`, `lr` etc. from the saved groups. Resuming an
  arm with a changed `split` or `muon_lr` silently keeps the old values (and `_plan` is rebuilt with the loaded `split`).
  Harmless as long as arms are never resumed with other options.

### N5. FixedValLoss runs before the EMA update of the same step
- Lightning puts model callbacks (`configure_callbacks` -> SimpleEMA) after trainer callbacks
  (`callback_connector.py` `_attach_model_callbacks`: `all_callbacks.extend(model_callbacks)`), so at step k the EMA
  evaluated is the EMA after k-1 updates while the raw net is after k updates. Negligible at decay 0.9999; state it.

### N6. A re-run smoke logs into the old W&B run
- `train_arm.sh:27` fixes the W&B id `abl-$ARM` with `resume: allow`. `smoke_ablation.sh:77` deletes the run dirs but
  not the W&B runs, so a second smoke resumes `abl-smoke_muon` and logs steps 1..300 again under a run already at 300
  (W&B drops non-increasing steps). Also the segments of one arm append duplicate FixedValLoss rows to
  `val_loss.jsonl` after a resume from an older checkpoint (B4/S4).

(N1-N6 above are the nits; the `## Nits` heading was kept for structure.)

## Checked and found correct
1. Routing (`muon.py:32-46`): Muon gets exactly the 2D weights under `blocks.` and `text_refine_blocks.`: qkv_x, kv_y,
   proj, w12, w3, adaLN_modulation.0 (blocks) and qkv, proj, w12, w3, adaLN (text blocks) = 16x6 + 4x5 = 116 matrices,
   1113.6M of 1122.0M trainable params (test output on 2 A40s). AdamW keeps x_embedder (proj1, proj2), t_embedder,
   y_embedder, y_pos_embedding (3D), final_layer, every bias, RMSNorm weights. `pos_embed` has requires_grad=False and is
   filtered out by `filter_nograd_tensors` before the optimizer. No Embedding modules exist.
2. Part boundaries: FUSED_PARTS matches the model: qkv_x [3d,d] reshaped `(3, heads, hd)` along d_out
   (`JiT_T2I.py:79`), kv_y [2d,d] `(2, heads, hd)` (`:84`), text qkv [3d,d] (`:143`), w12 `.chunk(2)` (`swiglu.py:14`),
   adaLN `.chunk(6)` (`JiT_T2I.py:114,170`). All cut along dim 0 in equal blocks. ResBlock adaLN (3 parts) is not
   used by JiT_T2I.
3. Math (`muon.py:143,153,167`): `lerp_(bufs, grads, 1-beta)` = beta m + (1-beta) g; `g.lerp(m, beta)` =
   (1-beta) g + beta m (Nesterov as in Keller Jordan's muon_update); NS coefficients and normalization by the
   Frobenius norm; transpose for tall matrices; scale 0.2 sqrt(max(rows, cols)) per unit (per part when split);
   decoupled weight decay (1 - lr wd) applied once per param before the update. With wd=0 the `if wd` skips it.
4. dtypes: O is bf16, `add_(upd, alpha=...)` into fp32 params promotes (no bf16 rounding of the weights); momentum
   is fp32 (zeros_like fp32 param).
5. Distributed: plan is deterministic (sorted by cost then index; ties by rank index), built from shapes only, so
   identical on all ranks; offsets disjoint; `all_gather_into_tensor` with equal shard sizes (`max(offsets)`, padded
   zeros); every rank applies all units. Test: 2 ranks identical, 2 ranks vs 1 process max diff 4.5e-8 (fp32 NS).
   The momentum is updated on every rank from the same averaged grads, so it stays identical. Grads that are
   bucket views (`gradient_as_bucket_view`) are only read (`lerp_` writes the buffers, `g.lerp` returns a new
   tensor), never written in place; clip_grad_norm_ scales them in place, which is the same as for AdamW.
6. AdamW part equals torch.optim.AdamW(fused=True) bit for bit (test: AdamW max diff 0.0 over 3 steps); the
   `_adam_fn` call matches torch 2.7.1's signature (`decoupled_weight_decay=True`).
7. Gradient clipping (`lightning_model_gpic.py:117-125`): global 2-norm over all params of all groups incl. the
   Muon group, before `optimizer.step`. Muon itself is invariant to the gradient scale except through momentum
   mixing, so clipping affects Muon only through the ratio of momentum and fresh gradient; same in the reference.
8. LR scheduler: one LambdaLR over all groups; the Muon group warms up with the same lambda.
9. FixedValLoss (`val_loss.py`): loss equals `JiTTrainer._impl_trainstep` with the same x, y, t, noise: per level
   1.738736 / 4.439307 / 444.142471 vs 1.738739 / 4.439317 / 444.140381 (bf16 autocast both; `numerics_check.py`).
   t convention: alpha(t) = t is the clean weight (LinearScheduler), x_t = t x + (1-t) noise, as in training. Levels
   are quantiles of sigmoid(N(-0.8, 0.8)) (t = 0.12 ... 0.61) with timeshift 1.0: correct for `lognorm_t: true`.
   Noise is seeded per image index j, independent of world size; rank slices are contiguous and disjoint; sums
   and the count are all-reduced in fp64, every rank calls all_reduce (no rank-0-only collective). EMA stream joined
   before the EMA forward. Captions padded to batch 32 for the compiled text encoder; uncondition not used
   (no CFG dropout), as stated. `val_transform` (center crop) is the val transform of GPICWebDataset(is_val).
   Eager `net.forward` on the compiled module calls the original forward (OptimizedModule.forward is the compiled
   one; `module.forward` here is the bound method of JiT_T2I because `.compile()` was called in place), so no
   recompiles. Logging with `log_metrics(step=global_step)` adds `trainer/global_step`, consistent with Lightning.
10. Configs: `lite_ablation.yaml` differs from `lite_ref_e1.yaml` (main) only in: DDP options, run name/id,
   checkpoint cadence (2.5k + ms 10k), FixedValLoss in place of SaveImagesHook, max_steps 20k, no validation
   (limit_val_batches 0, no sanity steps), no progress bar, fused AdamW, sample_every 0, exp tag. Model, data, batch
   32 x 8, WarmupFlatCooldown(4000, 4000, 400000), betas (0.9, 0.95), eps 1e-8, wd 0, clip 1.0 norm, EMA 0.9999,
   null_condition_p 0.1, seed 0 all kept. `lite_ablation_muon.yaml` only swaps the optimizer; the merged config
   parses with `--tags.exp` (print_config). `lite_sample.yaml` (eval) matches the ablation's model section except
   sampler steps 50 / CFG 1.0 (the eval protocol) and the optimizer settings (unused in predict).
11. `eval_arm.sh`: fails cleanly when no ms checkpoint exists; deletes only the npz it produced.
12. env.sh TMPDIR trap only removes its own job dir.

## Test coverage gaps (tools/test_muon.py)
The tests pass (2 ranks, A40, fp32 NS: ALL OK). They do not cover:
1. Save/load through a real checkpoint (torch.save + CPU map_location + `_optimizer_to_device`): misses B2.
   They deep-copy the state dict in memory.
2. Lightning's checkpoint path at all (two CheckpointHooks, last.ckpt contents, version counter): misses B1, B4.
3. bf16 Newton-Schulz (the production dtype); every exactness check uses fp32. No check of low-rank inputs, where bf16
   and fp32 disagree (S1), and no check of the per-group update RMS on realistic gradients (adaLN).
4. DDP with real `DistributedDataParallel` grads (bucket views, `gradient_as_bucket_view=True`), clipping, and the LR
   scheduler; they set `.grad` by hand.
5. The `p.grad is None` path (Muon uses zeros, so momentum decays and a frozen-for-a-step matrix still gets an
   update from momentum; AdamW skips): never hit, since every param gets a grad.
6. A world size that does not divide the work evenly and more units than ranks per shape (8 ranks); only 2 ranks.
7. `torch.compile` interaction and the full-size model on GPU (only meta-device tagging).
8. The job scripts (B3 would have shown in a 10-second `--print_config` dry run).
