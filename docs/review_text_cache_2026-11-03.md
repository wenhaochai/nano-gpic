# Review: text encoder outputs read from disk (git diff 076d651 HEAD)

Verdict: no blocker in the training math (association, T rule, masks, null caption, CFG, checkpoints). Fix S1-S4 before the smoke (S3 also before the sweep): as written, the smoke can pass while samples are silently dropped, and it can be killed at its time limit before it reports.

## Findings

### S1 (should-fix) jobs/smoke_ladder_cached.sh:2-5,39-53: the arms can need more than the 90-min --time, so arm 5, the final summary and `exit $fail` never run
Wall time of comparable arms in earlier runs (phases.log): d640 600 steps 6.7 min (ladder_prof_1102_2209 base_d640),
d1024 400 steps 7.4 min, d1280 400 steps 7.3 min, the epoch arm 2400 steps 12.6 min (ladder_smoke_1102_1636), eval 8.8 to
9.9 min (`predict_s` 895 to 1036 s plus 80 s fast_eval). Each launch has about 3 min of compile warm-up plus start-up.
The resume arm adds about 5 min. Sum of the required part: 6.7+7.4+7.3+5+12.6+9.5 = 48.5 min. That leaves 41.5 min for
d640_cpp (600 steps, about 7 min with a cold C++-wrapper compile, often more) and d640_graphs (1200 steps, cold
reduce-overhead compile), each capped at timeout 900 = 15 min. The total is near 78 min with no slack. One slow data
start breaks it: ladder_smoke2_1102_2116's d1280 arm waited 320 s of its 362 s for data (max 15 s per step). Each arm's
cap is 15 min, so the worst case is 7 x 15 + 15 = 120 min. At the limit Slurm (or the host's `timeout`) kills the
script: the python summary and `log DONE` do not run, and the exit code is 124/143, not `$fail`. The comment "Last, so a
stop at the time limit costs nothing else" is false, because the summary is after arm 5.
Fix: run `ladder_stages.py` and the python summary before the two measuring arms (or in a trap), and set --time 2:00:00.

### S2 (should-fix) jobs/smoke_ladder_cached.sh:55-68: the checks that matter for the cache only print. They do not set `fail`. A crash in the first check hides the rest
The heredoc prints "resume: first timed step", "same shards" and the count of `KeyError|exception|Traceback` in
epoch.log. No line sets `fail`, and the exit code of `python ... | tee` is never read. A key with no cache entry raises
in `_attach_emb`, and `wds.warn_and_continue` drops the sample (gpic_dataset.py:224, :337). So the smoke can drop every
sample of a shard whose cache is missing or stale and still exit 0. That is the class of failure this change brings in.
The heredoc also opens `bench/resume/steps_rank0.json` first. If the resume arm failed, the open raises a Traceback, and
the epoch check after it never prints. Fix: in python, `sys.exit(1)` when the first step is not 320, the epoch passes
are fewer than 3 or not the same, or the drop count is above 0. Wrap each check so one missing file does not hide the
others. Then fold the python's exit code into `fail`. Also read the shard-switch and pass counts from all 8 ranks' logs.

### S3 (should-fix) src/data/dataset/gpic_dataset.py:224-240,337: a missing or partial cache drops samples with a warning; training does not stop
`_attach_emb` raises on a missing `.idx.npz` or `.emb` file (np.load / os.open) or on a key with no entry, and
`wds.map(..., handler=wds.warn_and_continue)` drops the sample. Scenario: training starts before jobs/encode_captions.sh
has finished (the encode job has not run yet: data/gpic/train_emb_qwen3_1.7b does not exist on 2026-11-03), or one
shard's files are lost later. Each missing shard then drops 12.6K samples per pass, with one UserWarning per sample
in a 32-worker log, and the run trains on a different data mix with exit 0. If every shard is missing, every worker
yields nothing. The previous reviewer's warn_visible test shows the warning `KeyError('UPPER')`, then the stream goes
on. tools/check_text_cache.py proves the cache complete only inside the encode job. Nothing at train time checks it.
Fix: in `GPICWebDataset.__init__` (or `CachedTextConditioner.__init__`), when emb_dir is set, assert that every shard
of the expanded tar_pattern has both files (801 stat calls). In `_attach_emb`, let a failed open or a missing key
raise past the handler (for example, raise a class that the handler re-raises), so a hole in the cache is fatal. Also
add `--after ok:<encode wq id>` to the smoke submit.

### S4 (should-fix) jobs/smoke_ladder_cached.sh:30,48: the epoch arm runs under `timeout 900`, and it took 12.6 min with no cap in the last smoke
The same arm (d640, 16 shards, 2 workers per rank, 2400 steps) ran 17:29:54 to 17:42:32 in ladder_smoke_1102_1636
(phases.log). Its bench shows 153 s of data wait in a 623 s span: it is already bound by data. The cached path adds,
in only 2 workers per rank, one pread, one bytearray copy and about 6.7 MB of collated text per batch (32 x T 51 x
4 KiB) for the worker-to-main IPC. If the arm slows by 20% it reaches the 15-min cap: rc 124, `fail=1`, and the third
pass (the shard-switch check this arm exists for) never happens. Fix: no cap on this arm, or `timeout 1500`.

### N1 (nit) jobs/encode_captions.sh:15,22-23 and jobs/smoke_ladder_cached.sh:21: the default worktree gpic-wt/ladder_cache does not exist (checked 2026-11-03), and neither script checks it
Neither script uses `set -e`. `git -C $WT rev-parse` and `cd $WT/baselines/PixelGen` fail, and the script goes on. In
the encode job all 8 python processes then fail, so it exits 1 with no output. Create the worktree at the reviewed
commit before either submit, and add `[ -d $WT ] || exit 2` at the top of both scripts.

### N2 (nit) tools/encode_captions.py:100-103: a kill during a write leaves `<shard>.emb.tmp<pid>` (about 2.1 GiB) and `.idx.npz.tmp<pid>.npz`. No later run removes them
The host stops each segment of the job (fetchable-partial: timeout TERM, then KILL after 300 s). A process killed
during `tofile` leaves its temp file. A rerun picks a new pid name, so the old file stays. Each cut segment can leave
up to 8 x 2.1 GiB. `du` counts these files as "already written" (encode_captions.sh:20), so the free-space guard
overstates the room left. check_text_cache.py does not notice them. This is not a correctness problem, because the
rename makes a shard visible only when it is complete, and a shard with only the `.emb` renamed is encoded again.
Fix: at start, rank 0 removes `$OUT/*.tmp*` (no other run writes into OUT at the same time).

### N3 (nit) jobs/encode_captions.sh:27: when the job does not see exactly 8 GPUs, the script falls back to physical ids 0-7, not its own
`[ ${#GPUS[@]} -eq 8 ] || GPUS=(0 1 2 3 4 5 6 7)`. The job asks for gres gpu:8, so fetch_pending gives it only 8 free
GPUs, and a 4-GPU block cannot take it. Today the fallback is dead code. If it ever runs (for example, when
CUDA_VISIBLE_DEVICES is unset under some launcher), it places the processes on GPUs that belong to other guests.
Exit 98 there instead.

## Checked and found correct (by reading; the previous reviewer's tests are cited where they cover the point)
- Association: `_attach_emb` looks up `__key__` in the index of its own `__url__` shard. Each row's bytes go into
  `emb[r, :n]` of the same sample in `_collate_cached`. Nothing after `_group` reorders fields across samples. Covered
  by the exact fake-encoder test (9 shards, 3 workers, 2 epochs, 0 wrong rows) and by the real-data distances.
- Stale descriptor fix (HEAD): the state is reset to None before the opens. After a failed index load or a failed open,
  `_emb_shard` is None, so the next sample retries, and `os.close` never runs on a closed number. A worker that sees
  the same shard again in the next pass reuses a valid fd. The fd is opened in the worker, not before the fork.
- Equivalence: `text_len` is the length of the index (truncation 320, empty caption -> eos), which equals
  `len(input_ids)` of the old path. Same sort key in `_group`, same T rule (`_collate_cached` = `_collate_grouped`
  :289-290), the same mask. The encode tool uses the same tokenizer class, padding side, pad_multiple 8, sdpa,
  bf16 autocast, and the compiled `Qwen3TextEncoder._impl_condition` of masked mode.
- CachedTextConditioner vs Qwen3TextEncoder masked mode: `_to_cuda` is the same pinned, non_blocking copy (after
  Lightning's transfer, the tensors are already on cuda and are returned as they are). `y_mask = mask.bool()` is set
  the same way. The output is bf16 like the encoder's, so `BaseConditioner.__call__` casts nothing. The null row is
  written at position 0 and zeros after, of shape (len(y), T, 2048), as in the encoder (:76-85). The null is computed
  once by rank 0 from the same eos input. CFG dropout (training.py:15-25) and the RNG stream are unchanged.
- null.pt: only rank 0 writes it, with a pid temp file and os.replace. No encoder process reads it. There is no race.
- Shard files: each rank takes the disjoint shards `[rank::8]`. The skip needs both files. `.emb` is renamed before
  `.idx`, so a shard is skipped only after both are complete. The int16 view writes the same little-endian bytes that
  `torch.frombuffer(..., bfloat16)` reads.
- Checkpoints: `state_dict` saves the denoiser, the EMA and the trainer only. The conditioner has no parameters or
  buffers. Old last.ckpt loads strict (previous reviewer). Prediction keeps the online encoder (lite_ladder_sample.yaml
  :113-121), so a cached-path checkpoint loads there unchanged.
- compile_warmup: the cached branch feeds bf16 zeros [B, T, 2048] with the same half mask. A 12-step fit shows no
  recompile after the warm-up (previous reviewer).
- Smoke environment: a `VAR=x arm ...` prefix is exported to the arm's child processes and is cleared after it
  (tested: `X=1 f` gives 1 in a child, unset after). Each arm has its own LADDER_RUN_ROOT, so no files carry over. The
  d640 arm's MS = 2400/4 = 600 writes the ms-step 600 checkpoint that the eval arm reads. `timeout` kills the arm's
  process group (tested). The final python runs after a failed arm and does not set fail (S2).
- Memory: group_window 1024 x (196 KB image + 180 KB text) is about 385 MB per worker, x 32 workers x 8 ranks, about
  100 GB per node, within --mem 800G. Encode: up to 2 x 2.1 GiB per process per shard, x 8 processes.
- Free-space guard: avail 3688 GiB on 2026-11-03, so it passes now. free + written stays near constant during a run,
  except for the temp files (N2). Inodes: 1.17M free, and the cache adds 1602 files.

Not tested here: the encode job itself (it has not run; the cache directory does not exist), and a real 8-GPU cached fit.

## Fixes after this review
- S1: the summary of parts 1-4 runs before the two measuring arms, and --time is 2:15:00.
- S2: the checks set `fail`, each one in its own wrapper; a dry run on ladder_smoke_1102_1636 gives no false alarm.
- S3: `GPICWebDataset.__init__` asserts that every shard of the pattern has both files; `_attach_emb` runs with no handler, so a failed open, a key with no entry or a short read stops the run. tools/check_text_cache.py now compares the index's keys with the tar's json keys, because a key with no entry stops training. Test: tmp/text_cache_review/fatal_hole.py (real class, 3 workers): intact cache 192 of 192 rows and no error; each of the 4 holes raises.
- S4: the epoch arm has a limit of 1500 s; the eval has a limit of 1200 s.
- N1: both scripts exit 2 with no worktree. N2: the encode job removes `*.tmp*` before it reads the free space. N3: the encode job exits 98 when it does not see 8 GPUs.
