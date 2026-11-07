VERDICT: GO. The switch does what it says and is inert when unset. First create the worktree gpic-wt/ladder_seed at aaf2b67 (finding 1). Fix 2 if the result must be readable from a single pair.

Review of aaf2b67 (GPIC_RANK_SEED in host_tuning.py, jobs/smoke_ladder_seed.sh). Code read on 2026-11-03/04. CPU tests were not run (see the end).

## Findings, most severe first

1. **The worktree does not exist.** smoke_ladder_seed.sh:17-18 needs `$WC/project/gpic-wt/ladder_seed`. It is not in `gpic-wt/` (`git -C ../ladder_seed` fails). Failure: both wq jobs exit 2 at once. Fix: `git worktree add ../ladder_seed aaf2b67` (a frozen copy) before the submit.

2. **The two arms differ in more than the switch, and one pair has no noise estimate.** train_ladder.sh:41 sets `GPIC_SHARD_SEED=$(date +%s)` unconditionally. The two arms therefore see different shard orders, and the worker-level shuffle also differs (gpic_dataset.py:270 hashes the same seed). They also run on different nodes. Failure: an FD gap between the arms mixes the effect of the switch with data-order noise. One pair cannot separate the two. Fix (cheap): in train_ladder.sh:41, write `export GPIC_SHARD_SEED=${GPIC_SHARD_SEED:-$(date +%s)}`, and in smoke_ladder_seed.sh export one fixed value for both arms (e.g. 1730000000). Then the arms see the same data at each step, and only the noise/t/CFG streams differ. Production keeps its behaviour. Alternative: add a third job, `same` with another shard seed, to measure the noise floor directly.

3. **The job reports success even when an evaluation fails.** smoke_ladder_seed.sh:26-29: the loop echoes the rc, and the script's exit status is that of the last `echo`, i.e. 0. Failure: wq marks the job done with no metrics.json. Fix: `bash ... || fail=1` in the loop and `exit ${fail:-0}` at the end.

4. **A rerun overwrites the training log.** smoke_ladder_seed.sh:23 writes `> $ROOT/train.log`. After `wq requeue`, the first launch's log is lost (launches.txt in the run directory survives). Fix: `>>`. Also note that the W&B ids `seed-same`/`seed-rank` are fixed and `resume: allow` (wandb.py:344). A later repeat of the whole test appends to these runs unless the ids change.

5. **A resume replays the RNG stream from its start (both arms).** Lightning does not save or restore the RNG state. After a resume, `reset_seed` sets 0 again, the warm-up uses the same draws, and the switch sets the same per-rank seed again. So a resumed segment repeats the (t, noise, CFG) sequence of step 0 onwards, paired with new data. This holds in both arms and in production. It matters only if one arm resumes and the other does not. When you read the result, check launches.txt in both run roots.

6. **(Unverified) Exit 98 from the GPU health gate in a job that is not --partial.** train_ladder.sh:31 exits 98 on a bad GPU. smoke_ladder_seed.sh:25 passes it on. tools/wq/README.md defines 98 ("only waiting", requeue) and 99/124/143 in the paragraph on resumable jobs. I did not confirm what wq does with 98 or 99 for a plain job. If it marks the job failed, use `wq requeue wN`. That is safe (see Q3).

## Answers

### 1. Do the draws differ?
- Order: train_ladder.sh:38 lists ms-hook, resume-hook, CompileWarmup, HostTuning. Lightning's `_reorder_callbacks` moves only the ModelCheckpoint subclasses to the end and keeps the order of the rest (callback_connector.py:241-249). So HostTuning.on_train_start runs after CompileWarmup.on_train_start. Callback hooks run before the module's on_train_start (fit_loop.py:431-433).
- Seed value: the CLI calls `seed_everything(0, workers=True)` (cli.py:784; lite_ladder.yaml:2). DDP's `setup_distributed` calls `reset_seed()` on every rank (ddp.py:200, seed.py:72-82), which calls `torch.manual_seed(0)`. No other `manual_seed` exists in src/ or main.py (grep). So `torch.initial_seed()` is 0 on all ranks in on_train_start. The warm-up advances the generator but does not change initial_seed. Seed = 1000003·(rank+1): 1000003 … 8000024. These are distinct across ranks and the same at each launch of one rank (see finding 5).
- CUDA: `torch.manual_seed` calls `torch.cuda.manual_seed_all` (torch/random.py:46). This covers the rank's device.
- The draws: `torch.randn(..., device=x.device)` for t (training_JiT.py:39, lognorm_t true, yaml:149), `torch.randn_like(x)` (:43), and the CFG mask `torch.rand(..., device=condition.device)` (base/training.py:18) all use the default CUDA generator. They run eagerly: only `denoiser` and `ema_denoiser` are compiled (lightning_model_gpic.py:118-119). The trainer is not compiled. JiT_T2I has no dropout or other random op inside the compiled graph (the only randn is an init, JiT_T2I.py:274). So after the switch, each rank's t, noise and CFG mask differ.
- Without the switch, all ranks are identical at each step: the same seed 0 and the same warm-up draws (compile_warmup.py, B=32 on every rank). The claim "32 different (t, noise) per global batch" holds.
- Resume: on_train_start is reached on every `fit`, resumed or not (fit_loop.run → on_run_start, fit_loop.py:208-213, 414-433).
- Data workers: `setup_data` runs first and calls `iter(self._data_fetcher)` (fit_loop.py:209, 276). That starts the persistent workers and draws their `_base_seed` from the generator (dataloader.py:698). Both happen before on_train_start. The workers are persistent (gpic_dataset.py:416), so no later iterator draws again. The shard and group shuffles use their own `random.Random` objects seeded from GPIC_SHARD_SEED, the worker id and the pid (gpic_dataset.py:54, 270). Confirmed: the workers do not depend on the switch.

### 2. Does anything assume equal RNG across ranks?
- DDP all-reduces the gradients. It needs equal parameters and equal bucket order, not equal draws.
- CFG dropout with y_mask: `torch.where(mask[:, None], null_row, ym)` keeps the shape [B, T] (base/training.py:19-22). `condition*(1-mask)+uncondition*mask` keeps the shape. The mask is data, not a shape or a Python branch, so dynamo has no guard on it and does not recompile. T already differs between ranks, and the warm-up made T dynamic (compile_warmup.py docstring).
- EMA: each rank updates its EMA from the same parameters after the all-reduced step, with no RNG. Gradient clipping (lightning_model_gpic.py:122-130) acts on the all-reduced gradients, so the result is identical on all ranks. The SIGTERM broadcast (utils/patch_bugs.py:21-35) uses no RNG.
- Logging: `loss` is logged without sync (lightning_model_gpic.py:232), so the logged train loss is rank 0's batch, before and after the change. The per-caption-type losses are synced.
- Unset (production): the block runs only when `os.environ.get("GPIC_RANK_SEED") == "1"`. With the variable unset, the only code that runs is the existing gc.collect/freeze. No change.
- The variable is also exported into eval_ladder.sh. Predict does not use the training path, and its noise comes from per-prompt generators. I did not read lite_ladder_sample.yaml to confirm that HostTuning is absent there. Even if present, on_train_start is not called in predict.

### 3. The job script
- Checkpoint steps: MS = 66816/4 = 16704, so ms checkpoints fall at 16704, 33408, 50112 and 66816. LAST = 2088 (train_ladder.sh:27). 66816 % 32 = 0. The ms filename is `ms-step=066816.ckpt`. The exit check (train_ladder.sh:53) and eval_ladder.sh:20 (`ms-*033408*`) match it.
- Exit rule: with LADDER_PARTIAL_OK unset, a launch that ends with rc 0 but without ms-step=066816 becomes rc 99. smoke_ladder_seed.sh:26 then exits 99 and skips the evaluations. Correct.
- Time: 66816 × 29.8 ms = 33.2 min, plus 32 × 2 s saves, ~80 s start-up and the health gate: about 36 min. Two evaluations at ~10 min each: 20 min. Total about 56-60 min, against --time 1:40 (100 min). The margin is 40 min. The worst case at eval_ladder.sh's own 45 min limit would be 36 + 90 min and would not fit. That needs an evaluation 4.5× slower than expected.
- Arguments: fetch_pending reads wq's `args` file (NUL-separated) and runs `bash "$script" "${gargs[@]}"` (fetch_pending.sh:245, 297). `wq submit jobs/smoke_ladder_seed.sh same` therefore gives $1 = same.
- Overrides: `"$@"` comes after the default `--trainer.logger.init_args.name/id` (train_ladder.sh:49-50). jsonargparse keeps the last value, so the runs become seed_same/seed-same and seed_rank/seed-rank. The project is gpic-speed (lite_ladder.yaml:26; env.sh sets WANDB_PROJECT, WANDB_BASE_URL and WANDB_DIR).
- LADDER_WT and LADDER_RUN_ROOT are exported (smoke_ladder_seed.sh:22), so both train_ladder.sh and eval_ladder.sh see them.
- A stopped job: fetch_pending runs it under `timeout --kill-after=300 <s>` (fetch_pending.sh:297). A host that ends early sends a signal, and main.py's SIGTERM path exits. The run root is fixed per arm, so a `wq requeue` resumes from the newest `last*.ckpt` (train_ladder.sh:33-34). The ms hook saves before the resume hook (train_ladder.sh:37-38), so no last.ckpt gets ahead of a missing ms file. That is safe for a fetchable job. At most 2088 steps (~62 s) are lost, plus the RNG replay (finding 5). A job under 3 h may skip --partial (README rule).

### 4. Comparability and how to read the result
- Equal in both arms: code, config, seed_everything(0) before instantiation (the same initial weights), the warm-up, the LR schedule and the step count.
- Different: (a) the switch; (b) GPIC_SHARD_SEED, i.e. the shard order and the in-worker grouping order (finding 2); (c) the node, and bf16 nondeterminism in reductions; (d) any resume the arms do not share (finding 5).
- Noise: no FD repeat of an identical d640 run exists in the docs I read, so the noise floor is unknown. With 50K samples, the sampling part of FD-DINOv2 is small. The training-run part (data order, nondeterminism) at a short budget is often a few percent of FD. The expected effect is also modest: per-rank images already differ, and only the 8× repeat of (t, noise) is removed. Effect and noise may be of similar size.
- Read: compare both checkpoints (C0/16 and C0/8). A gain that has the same sign at both and is larger than a few percent is evidence. A gap of 1-2 % is "not resolved", not "no effect". To get a real answer, apply the fix in finding 2 (paired data order) or add one repeat arm. The train-loss curves are a weak extra check: the logged loss is rank 0 only in both arms.

## Checked and found correct
- The seed formula, its distinctness across ranks, the CUDA seeding, the hook order after CompileWarmup, and that the hook is reached on resume.
- Data workers start and draw their seed before the switch. Their shuffles use their own generators.
- No shape change and no recompilation from per-rank CFG masks. EMA, gradient clipping and the SIGTERM broadcast do not depend on the RNG. Production is unchanged when the variable is unset.
- The ms steps 33408 and 66816 exist with matching file names. The rc 99 rule applies. The W&B name and id overrides work. The env vars reach eval_ladder.sh. wq passes $1. --time has a 40 min margin at the expected speed. Resume from the fixed run root is safe.

## Not checked
- No CPU test was run. Claude Code's temporary volume filled up (ENOSPC) partway through the review, and no command could start after that. All conclusions come from reading the code: Lightning 2.6.6 and torch 2.7.1 in envs/pixgen, the repo at aaf2b67, and tools/wq and fetch_pending.sh.
- wq's handling of exit 98/99 for a job that is not --partial (finding 6).
- lite_ladder_sample.yaml's callback list.
- Whether 29.8 ms per step and 10 min per evaluation hold on the node the job lands on.
- A measured FD run-to-run noise for d640.

## Fixes (2026-11-04)

- 1: the worktree gpic-wt/ladder_seed is created at the fixed commit before the submit.
- 2: train_ladder.sh keeps a GPIC_SHARD_SEED that is already set; the test job sets one value for both arms, so they read the same data in the same order. A launch of production sets none, so it behaves as before.
- 3: the job exits non-zero when an evaluation fails.
- 4: the training log appends.
- 5, 6: no change; launches.txt is read with the results.
