# Review: branch ladder-full, 1bee608..cb33b6b (and the commits after it)

Reviewer: independent adversarial review, 2026-11-06, read-only.
Worktree: `gpic-wt/ladder_full`. Branch HEAD at review time: 108afa1.

## 0. Scope note

The requested range ends at cb33b6b. Commit eda7e67, made after the range, changes the core design. The log now records the shards a worker *starts* (`shards_started_rank<r>.txt`), not the shards it *finishes* (`shards_done_rank<r>.txt`). The smoke w1313 ran eda7e67, and the production code will be HEAD. So I reviewed both versions:

- the finished-log design in cb33b6b (finding B0);
- the started-log design at HEAD (all other findings).

Lightning 2.6.6, webdataset 1.0.2, torch from `envs/pixgen`.

Evidence I produced (CPU only, vis1):

1. `tools/test_shard_resume.py` at HEAD: ALL OK (6 of 6). `tools/test_ladder.py`: ALL OK (items 1, 2, 5, 6, 7).
2. A sample-level probe through the real path. Script: `tmp/ladder_prof/review_probe/probe.py`, `probe_drop.py`. It covers: `_shard_stages`, `wds.tarfile_to_samples`, `wds.shuffle`, and a torch DataLoader with 3 forked persistent workers and prefetch_factor 4. It uses 24 synthetic tars of 50 samples, world 2, and both ranks. The steps:
   1. train 25 batches;
   2. take a "checkpoint" time T;
   3. read 10 or 40 more batches and throw them away;
   4. stop;
   5. apply the `train_ladder.sh` filter (time <= T);
   6. resume.

   Result, both ranks:
   - 0 duplicate samples in pass 0;
   - with 3 log lines dropped, the dropped shards are read again, and no kept shard is read again;
   - pass 1 has 600 distinct samples (the whole rank).
3. Smoke w1313 (`runs/gpic-speed/full_smoke_1106_1020`): `check_shard_resume.py` ALL OK. See S2 for what this smoke does not exercise.

---

## Blocker

### B0 (in the requested range, cb33b6b only; fixed at eda7e67). The finished-shard log trains the open shards twice on every resume.
- Where: `gpic_dataset.py` `_log_finished` at cb33b6b, and `train_ladder.sh` at cb33b6b.
- Scenario: at the checkpoint, each of the 32 workers per rank has one shard half read. Those shards are not in the finished log. A resume reads them again from the start, so their consumed head is trained a second time in the same pass. That is about 0.5 shard × 32 workers × 8 ranks ≈ 128 half-shards per resume, or 0.8 % of GPIC Full trained twice per resume.

  The other direction loses data. A shard is logged "finished" when the worker has pulled its last sample. Up to about 1 250 of its samples (group window 1 024 + shuffle buffer 100 + prefetch queue 128) can still sit in the worker buffers, not yet trained at the checkpoint. The resume skips them.
- Fix: done in eda7e67 (log at start). Do not deploy cb33b6b. Create the frozen worktree `gpic-wt/ladder_full_run` at eda7e67 or later; it does not exist yet. The default `WT` of `train_full.sh` and `train_ladder.sh` points at it.

No blocker at HEAD.

---

## Should fix

### S1. The shard state lives outside the checkpoint, and the cut is the file mtime of last.ckpt.
- Where: `jobs/train_ladder.sh:45-54` (`shard_seed.txt`, `stat -L -c %Y "$LASTCK"`, `awk '$1 <= T'`).
- In the normal flow this is correct. Every line kept has a start time ≤ mtime. Every sample consumed by the checkpointed step came from a shard that started before the dump began, and the dump began before the mtime. The integer floor of `%Y` only drops more lines. I checked that fsspec's transaction (mkstemp, then `shutil.move` in the same fileset) keeps the mtime of the last write. No duplicate is possible as long as the mtime is the true save time and the logs and the seed stay with the checkpoint.
- Failure scenarios:
  1. **Continue from a copy.** For example, the Lite 4C0 continuations copy `last.ckpt` into a new run dir (`continue_run.sh`). If someone copies only `last.ckpt`, the new dir has no `shard_seed.txt` and no logs. The run gets a new seed and starts pass 0 of a new permutation, so it reads again data it has already trained on. This is silent.
  2. **A copy with a new mtime.** Plain `cp`, or `rsync` without `-t`, gives the file a new mtime. Then `KEEP_T` covers every line written after the real save. All shards started in the up to C0/8 steps after the checkpoint are skipped and never trained in that pass. For d640, C0/8 = 66 800 steps × 256 images / about 12 600 images per shard ≈ 1 360 shards, which is 17 % of the pass.
  3. **An older mtime.** A restore with `cp -p` of an older file, or clock skew larger than the delay from a shard's start to its first consumption, makes the reverse error: duplicates.
- Fix (cheap):
  1. Put the cut in the checkpoint. In the CheckpointHook (or a small callback), store `time.time()` in `on_save_checkpoint` of `last*.ckpt`, together with the shard seed.
  2. On load, export the stored time and seed (rank 0 reads the ckpt, or `train_ladder.sh` reads it with `torch.load(..., mmap=True)` of the callback key). Filter by the stored time instead of the mtime.
  3. Assert that `shard_seed.txt` equals the seed in the checkpoint.

  The dump time is also a tighter cut than the mtime. The mtime is 30–60 s later for d1536/d1792, and during that save every worker keeps prefetching and may start a shard that is then lost. At minimum, document that a run is moved or continued only with `cp -p` of `last.ckpt`, `shard_seed.txt` and `shards_started_rank*.txt` together.

### S2. The smoke does not exercise the mtime filter or the "no sample twice" claim.
- Where: `jobs/smoke_full.sh:47` (`LADDER_LAST_EVERY=200 … --trainer.max_steps 1300`) and `tools/check_shard_resume.py:72` (check 1b).
- What happened in w1313: launch 2 "kept 32 lines … and dropped 0 later ones". With 2 workers per rank, a worker opens a new shard about every 790 steps (steps ≈ 0 and ≈ 790). The window between the checkpoint (step 1200) and the stop (step 1300) had no shard start. So the drop path of the filter, the only part that decides between a duplicate and a loss, never ran on GPU.

  Check 1b uses the same `keep_t` that the bash filter used, so it is circular. It cannot catch a wrong cut. Checks 1c and 1d compare shards *started*, not samples *trained*.
- Fix:
  1. Use `LADDER_LAST_EVERY=500 --trainer.max_steps 1000`. The checkpoint falls at step 500, and the shard starts at about step 790 fall after it, so they must be dropped and read again.
  2. Require `n_dropped >= WORLD` in 1b.
  3. Check that `keep_t` ≥ the time of the last kept line and < the time of the first dropped line.
  4. Optionally log the `__key__` of each trained sample on one rank for launch 1 up to the checkpoint step and for launch 2, and assert no key appears in both.

  My CPU probe (evidence item 2) covers the sample-level claim and the drop path through the real DataLoader, but not Lightning's save timing.

---

## Minor

### M1. `_rank_world` falls back to an environment that Lightning sets only in part.
- Where: `gpic_dataset.py:42-47`.
- Lightning's subprocess launcher sets `WORLD_SIZE`, `LOCAL_RANK` and `NODE_RANK`, but not `RANK`. If the train dataset were ever built before `init_process_group`, every rank would get (0, 8). All ranks would then read `perm[0::8]` and append to `shards_started_rank0.txt`. The `__iter__` assert would not catch it, because the forked workers see the same fallback.
- This is not reachable today: `train_dataloader()` runs in `fit_loop.setup_data`, after `strategy.setup_environment()`. The smoke log shows 8 different ranks.
- Fix: if `WORLD_SIZE` > 1, dist is not initialized and `RANK` is not set, raise.

### M2. Every resume loses the unread rest of the open shards, and the segment boundaries also count as resumes.
- Where: design of `_log_started`.
- Cost: about 0.5 shard × 32 workers × 8 ranks ≈ 128 shards' remainders (1.6 % of the 8 000 shards) per resume, as `docs/ladder_plan.md` 12.4 says. The tail before the dump (S1) adds to it.
- Effect by size:
  - Under 1 epoch (d1792 0.73, d1536 1.10): there is no repetition. The pass just reads further into the permutation, so this is harmless.
  - Multi-epoch sizes (d1024 3.25 epochs): those samples are seen one time fewer.
- Accept, or remove it by storing per-worker consumed-sample counts. That is more work and not needed for correctness.

### M3. `GPIC_SHARD_SEED` from the environment bypasses `shard_seed.txt` and never writes it.
- Where: `train_ladder.sh:45-48`.
- Scenario: a launch with the variable set starts the logs with that seed. The next launch without it draws `date +%s`. `_resume_point` then raises `assert done <= mine`, so the job fails at every retry until wq's run limit. It fails loudly, but it blocks the chain.
- Fix: always write the file if it is missing, and refuse a mismatch between the environment and the file.

### M4. A missing worktree passes the "no local changes" check.
- Where: `train_full.sh:26` and `smoke_full.sh:25`.
- `git -C <missing dir> status --porcelain` prints only to stderr, so `[ -z "" ]` is true. `exec bash $WT/jobs/train_ladder.sh` then exits 127. wq counts 127 as a failure, so the `after ok` chain stops.
- Fix: add `[ -d "$WT/.git" ] || [ -f "$WT/.git" ] || exit 2` first. Note: `gpic-wt/ladder_full_run` does not exist yet.

### M5. `last.ckpt` saves cost 4–7 % of wall time for the two new sizes.
- Smoke w1313: a save stalls 36.6 s (d1536) and 57.8 s (d1792).
- LAST = C0/8 is 6 800 steps × 136.5 ms = 928 s for d1536, and 4 500 steps × 193 ms = 868 s for d1792. The save overhead is therefore 3.9 % and 6.7 % of wall time.
- FLOPs do not change, only GPU-h (about 10 GPU-h for d1792 to 8C0).
- Optional: use C0/4 for d1536/d1792, or a faster or asynchronous save.

### M6. The cache check assumes every json of the 7 199 new shards has a caption.
- Where: `encode_captions.sh:44` → `tools/check_text_cache.py` (cache_full / ladder_cache).
- `encode_captions.py` indexes only the json files that have both `caption` and `caption_type`. The check compares the index with *all* `.json` keys. One json without a caption in any new shard makes the cache job exit 1 forever, and the training segments (`after ok`) never start.
- I sampled 3 new shards (02352, 02360, 05000): 0 such json files. Low risk, and it fails loudly.
- Fix: make the check use the same caption filter.

### M7. Each size has its own shard seed.
- Each run draws `date +%s` at its first launch, so the 7 widths read different sample orders.
- For a width-vs-compute ladder, one fixed seed for all sizes removes the data-order variance from the comparison at no cost. Every width would read the same prefix of the same permutation, apart from the per-resume losses in M2.
- Suggest: write the same value into each `exp_ladder_<size>/shard_seed.txt` before the first launch.

### M8. A torn log line stops the resume.
- Where: `gpic_dataset.py:93` (`assert all(len(r) == 3 …)`).
- A process killed inside the one-line `os.write` would leave a partial line. Every later launch then fails on the assert. It is very unlikely, because the write is one short O_APPEND syscall.
- Optional: skip a malformed *last* line with a warning instead of asserting.

### M9. The unit tests cover only the environment path and none of the bash.
- `tools/test_shard_resume.py` and `test_ladder.py` item 6 set `RANK`/`WORLD_SIZE` (the M1 fallback), not the `torch.distributed` path that production uses. They do not test the `awk` time filter or the mtime.
- The tests do test what their docstrings claim: the split, log content, skip, next pass, foreign seed, and that validation has no log. The production path is covered only by the smoke, with the gap described in S2.

---

## Checked and found fine (HEAD)

1. **Rank and world at construction.** `train_dataloader()` is called from `FitLoop.setup_data` after `DDPStrategy.setup_environment()` has initialized the process group, so `_rank_world()` uses `dist`. Workers are forked (Linux default), so `dist.is_initialized()` and `get_rank()` work in each worker, and the `__iter__` assert holds. w1313 logs show 8 distinct ranks and one log file per rank.
2. **One pass per epoch per rank, and the log read before the fork.**
   - `iter(WebLoader)` returns a generator (`DataPipeline.__iter__` → `iterator()`). Lightning's `_check_dataloader_iterable` therefore starts no workers.
   - The first `next()` of the prefetch fetcher creates the persistent workers, and each worker calls `iter(dataset)` once.
   - Each later epoch sends `_ResumeIteration`, which is one more `iter()` per worker.
   - On a resume, `TrainingEpochLoop.on_run_start` skips its extra `iter()` because `fit_loop.is_resuming` is True (also with `current_epoch > 0`).
   - The Lite d1280 log shows exactly passes 0, 1, 2 for 2.27 epochs, and w1313 shows pass 0 then pass 1 once each after the resume.
   - The log is read in `EpochShuffledShards.__init__`, in the rank's main process, before any worker of this launch exists. The previous launch has ended, so there is no race with appends.
3. **Ranks end their passes at different steps.** No collective runs at a training epoch end:
   - the train metrics are `on_step` only;
   - `ModelCheckpoint.on_train_epoch_end` skips because `every_n_epochs` = 0 and `_last_global_step_saved == global_step`;
   - no callback hooks `on_train_epoch_end`.

   So a rank that runs out simply starts its next pass while the others continue, as in the Lite runs. The pass of each rank comes from its own log, so it does not depend on Lightning's `current_epoch`, which is restored from rank 0's checkpoint.
4. **Node and worker split.** `perm[r::world]` minus the skip set, then `wds.split_by_worker`, is the same as the old `split_by_node` → `split_by_worker`. All worker copies are forked from the same object, so they hold the same skip set, and `skip` is cleared after the first pass in each copy. The remaining list is split without overlap. Test 3 and my probe confirm this.
5. **Resume point.**
   - A rank's DataLoader starts pass p+1 only after the main process has drained pass p, so "the newest pass in the log" is right.
   - "All shards of pass p started → pass p+1" is right.
   - A changed seed, shard list or GPU count trips `done <= mine` loudly.
   - A changed number of workers is fine.
6. **Log write point.**
   - `url_opener` pulls the next url only when `tar_file_expander` asks for the next source, so the line is written when the worker starts the shard.
   - A missing tar is logged and then skipped by `warn_and_continue`, which is harmless.
   - Each line is one O_APPEND write per worker. No line is mixed in the logs I read.
7. **Validation set and other instances.**
   - `is_val=True` gets no log, and `limit_val_batches: 0` means it is not built.
   - The tools that build `GPICWebDataset` (`test_text_cache`, `test_worker_exit`, `datapath_prof`) do not set `GPIC_SHARD_LOG_DIR`.
   - The unseeded path still uses `SimpleShardList` + `split_by_node` + `split_by_worker`.
8. **mtime semantics.**
   - The `_atomic_save` → fsspec transaction writes a temp file in `$TMPDIR`, which is in the same GROUP fileset, and then renames it. The mtime is the end of the write, which comes after the dump, which comes after the last consumed batch.
   - The floor of `%Y` is the safe direction.
   - The log times and the mtime come from the same node. On vis1, a probe showed the mtime between `time.time()` before and after the write.
   - `stat -L` is right if `last.ckpt` is ever a link.
   - Lines from a launch that saved no new checkpoint have times later than the old mtime, so they are dropped and those shards are read again.
9. **Bash.**
   - A glob with no match is guarded by `[ -f "$f" ]`.
   - `awk -v T=<int> '$1 <= T'` compares numbers: both sides are strnum.
   - `awk … > tmp && mv || exit 2` is correct.
   - `pipefail` without `-e` is harmless here: the `ls | head` assignment and the explicit `rc` checks.
   - The later `--trainer.logger.init_args.{name,id}` override the defaults (jsonargparse takes the last value).
   - `exec` in `train_full.sh` passes the exit code to wq.
   - `[ "$LAST" -gt 0 ]` guards a bad `LADDER_LAST_EVERY`.
10. **Budget segments.**
    - `STEPS = S·M` with S a multiple of 800 is a multiple of 32.
    - `MS = S·M/4` gives ms saves at {C0/4, C0/2, 3C0/4, C0}, {1.5, 2}C0, {3, 4}C0 and {6, 8}C0. The steps that a resumed segment has already passed are not saved again.
    - `LAST = S/8` divides every MS and the final step, so the last step always writes `last.ckpt`.
    - `LAST` is the same in every segment, so the last-hook's `state_key` matches and `last_model_path` is restored: `last.ckpt` is overwritten, with no `last-v1.ckpt`. The ms-hook's key changes per segment, so its state is not restored, which is harmless.
    - The exit-0 check `ms-step=$(printf %06d $STEPS)` matches Lightning's `ms-{step:06d}` also for 7-digit steps (for example 1 286 400).
    - The W&B id `ladder-full-<size>` with `resume: allow` continues one run, separate from the Lite ids.
    - The LR stays flat (`WarmupFlatCooldown`, max_steps 1e9).
    - No `on_exception` save (`save_on_exception=False`), so a SIGTERM leaves the previous `last.ckpt`.
11. **Lightning loader state.** WebLoader has no `state_dict`, so the checkpoint stores no `combined_loader` state that could fight the shard log.
12. **encode_captions.sh guard.** `df` on the fileset reports the fileset quota (125 T size, 37 T free now). The guard of free + written ≥ 20 TiB is consistent with a 16.6 TiB cache and leaves about 3.4 TiB. The encoder is running: 2 369 of 8 000 shards indexed, rank 0 at 296/1 000.
13. **`check_shard_resume.py` part 2.** It gives the correct step-time and memory table (d1536 136.5 ms, 44.9 GiB reserved; d1792 193.0 ms, 61.2 GiB). `test_ladder.py` item 7 now tests the `text_len` field, which both data paths set.
