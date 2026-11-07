VERDICT: NO-GO — the job runs the code of worktree gpic-wt/ladder_kern, which is at 89dbf49 and has neither switch, so 4 of its 9 arms would silently repeat the control; fix the worktree (finding 1), then GO.

Review of 0c94af4 (GPIC_SIGTERM_EVERY, GPIC_WORKER_NICE, jobs/smoke_ladder_host2.sh), read against Lightning 2.6.6 and torch in
envs/pixgen (python3.11). Nothing was run on a GPU, nothing was submitted. Paths of installed packages are relative to
envs/pixgen/lib/python3.11/site-packages. Line numbers in the repo are at 0c94af4.

## Findings

### 1. jobs/smoke_ladder_host2.sh:16 — the code worktree does not contain the commit under review (blocker)
`WT=$WC/project/gpic-wt/ladder_kern`. `git worktree list` shows ladder_kern at 89dbf49 (detached). In that tree,
`grep GPIC_WORKER_NICE\|GPIC_SIGTERM_EVERY` finds nothing: patch_bugs.py still has the fixed `% 50`, and HostTuning has no nice
code. train_ladder.sh runs `$WT/baselines/PixelGen/main.py` (train_ladder.sh:22, 41). So d640_nice10, d640_w16_nice10,
d640_bcast200 and d1280_bcast200 run as production. No log line is missing loudly: the nice arms would only lack the `[nice]`
lines. The worker and bucket arms would still be valid, because they use CLI flags.
Fix: keep ladder_kern at 89dbf49 (it reproduces w1048). Add a new worktree at the reviewed commit, e.g.
`git worktree add --detach $WC/project/gpic-wt/ladder_host2 <commit>`, and set `WT=` to it in line 16. The commit that does this
then also carries the fixes below; point the worktree at that commit.

### 2. jobs/smoke_ladder_host2.sh:4, 46-64 — the arm limits sum to 68 min against --time of 40 min; the expected run is about 35 min
Limits: 6 x 420 + 420 + 720 + 420 = 4080 s. Expected times, from earlier jobs on the same code path:
- d640 with 2400 steps: 180-191 s (w1048 d640_pin 3:11, d640_cycle 3:00). Six arms: about 19 min. The w16 arms can be slower if
  16 workers per rank cannot keep up.
- d1280 with warm caches, 800 steps: about 240 s (w1048 d1280_prod: 275 s for 1200 steps, 70 s of warm-up).
- d1280_b3000, cold compile, 800 steps: about 430 s (w974 d1280_b1000, cold, 700 steps: 423 s; warm-up 98+92+80 s).
- gpu_health.py in each of 9 launches.
Sum: about 34-36 min. The margin is about 5 min. If the job reaches its --time, the host's `timeout` (tools/fetch_pending.sh:297)
stops the last arm, and d1280_b3000 (the arm with the largest expected gain, lever 2) is second to last.
Fix: `--time=00:55:00`, and run d1280_control and d1280_b3000 right after d640_control. Finding 3 frees about 7 min more.

### 3. jobs/smoke_ladder_host2.sh:61, 64 — the two broadcast arms cannot resolve their effect; one control does not give the noise
Expected gain from 50 to 200 steps: (1/50 - 1/200) x 30 ms = 0.45 ms per step at d640 (1.5%), (1/50 - 1/200) x 55 ms = 0.83 ms
at d1280 (0.9%). Launch-to-launch noise is at least that large: w974 measured the broadcast fully off at median 32.5 against
32.4 ms for the control (docs/text_cache.md:45). Each arm also reads other shards (GPIC_SHARD_SEED is the launch time,
train_ladder.sh:35), and step time depends on the batch's T (text_cache.md:50). The gain is pure arithmetic once the cost of a
broadcast step is known, and BenchTimer already writes per-step times (src/callbacks/bench.py:222, steps_rank<r>.json).
Fix: drop both broadcast arms. Compute the cost per broadcast step from the control arms: mean of steps with step % 50 == 0 minus
the others. Use the freed time for a second d640 control at the end of the job, so the d640 arms have a launch-to-launch spread.
If you keep the arms, keep them only as a check that the switch causes no hang.

### 4. src/utils/patch_bugs.py:30 and src/callbacks/host_tuning.py:94 — bad values crash every rank
- `GPIC_SIGTERM_EVERY=0`: `global_step % 0` raises ZeroDivisionError in the first `advance` of every rank.
- `GPIC_SIGTERM_EVERY=` (empty): `int("")` raises ValueError at import. The same holds for `GPIC_WORKER_NICE=` in
  HostTuning.__init__.
No arm of this job sets these values. A later job could.
Fix: `_SIGTERM_EVERY = max(1, int(os.environ.get("GPIC_SIGTERM_EVERY") or 50))` and
`int(os.environ.get("GPIC_WORKER_NICE") or 0)`.

### 5. Two switches can fail without a sign; check each in the logs before reading results
- GPIC_WORKER_NICE: 8 lines `[nice] device k: 32 of 32 workers at nice 10` (16 of 16 in d640_w16_nice10), and pt_data_worker
  at NI 10 in `$PROF/<tag>/nice.txt`. Any exception prints `[nice] skipped: ...`, and the arm runs as production.
- GPIC_SIGTERM_EVERY prints nothing. Add one line at import when the variable is set, e.g.
  `print(f"[sigterm] broadcast every {_SIGTERM_EVERY} steps", flush=True)`, or check the step % 200 pattern in steps_rank0.json.
- Worker count and bucket cap: the saved config in `$OUT/<tag>/exp_ladder_<size>/config-*.yaml`; for b3000 also a cold
  `[warmup] T=64` time (about 100 s, against 33 s warm).

### 6. Expectation, not a defect: DDPOptimizer keeps a 1 MiB first bucket, so the compiled graph still splits in two
The reducer gets one bucket (see "checked" below). Dynamo's DDPOptimizer is built with `bucket_bytes_cap` only
(torch/_dynamo/convert_frame.py:1413). Its `first_bucket_cap` then is `_DEFAULT_FIRST_BUCKET_BYTES`, 1 MiB
(torch/_dynamo/backends/distributed.py:384-390), and the split loop starts a new bucket once the first passes it (lines 441-444).
So each direction has 2 compiled subgraphs: the last layers (about 1 MiB of parameters) and the rest. The one all-reduce still
starts only after the whole backward: with one bucket nothing overlaps the backward. That is what the arm measures.

### 7. Low risk, watch the data wait: a niced worker serves the file descriptors of its batches
On Linux torch shares batch tensors with the `file_descriptor` strategy (torch/multiprocessing/__init__.py:56). The descriptors go
through `multiprocessing.reduction.DupFd` (torch/multiprocessing/reductions.py:619), served by a resource_sharer thread in the
worker. That thread exists before the first training batch (prefetch, factor 4), so nice_workers sets it to nice 10 too. The
rank's pt_data_pin thread (nice 0) then waits on a nice-10 thread. On contended CPUs that adds wait to the pin path. BenchTimer's
data wait (rank 0) and thread_wait's pt_data_pin line show it. GPFS reads are plain `pread` and tar reads in the worker
(gpic_dataset.py:258); no user-space lock is shared between workers, and the GPFS daemon is not niced.

## Checked and found correct

a. patch_bugs.py
- `import os` is at line 12, before its first use at line 30.
- main.py:6 imports the module at the top. Ranks 1-7 are started by Lightning with `subprocess.Popen(command, env=env_copy)`,
  where env_copy is a copy of rank 0's `os.environ` plus LOCAL_RANK (lightning/fabric/strategies/launchers/subprocess_script.py,
  the `env_copy` lines near 125-145). Each rank re-runs main.py, imports the patch, and reads the same value at import. The patch
  replaces the class attribute, so every loop instance uses it.
- The call site is `advance` (lightning/pytorch/loops/training_epoch_loop.py:299-300), before each batch, with `global_step` =
  optimizer steps done. accumulate_grad_batches is 1 (lite_ladder.yaml:87) and bf16-mixed has no grad scaler, so every rank steps
  at every batch, and `global_step` is equal on all ranks. On resume it comes from the same last.ckpt on every rank. So all ranks
  call the broadcast at the same global steps for every N ≥ 1, also after a resume at a step that is not a multiple of N (the
  first call is at the next multiple).
- KILL 45 s after TERM (tools/fetch_pending.sh:138): at N = 200 one interval is 17.6 s at d1280 (88 ms) and 5.4 s at d640.
  N = 500 would take 44 s at d1280: too close. More important, N matters less than it seems. Slurm and `timeout` send TERM to every
  rank. Each rank's handler sets `received_sigterm`, and `on_advance_end` raises SIGTERMException locally at the end of the current
  step (training_epoch_loop.py:421), with no collective, whatever N is. The periodic broadcast only matters when rank 0 alone gets
  the signal.
- If the KILL comes first, nothing is lost that N could save: no checkpoint is written on SIGTERM (CheckpointHook adds no
  on_exception save; ModelCheckpoint's save_on_exception defaults to False, lightning/pytorch/callbacks/model_checkpoint.py:271).
  The resume uses the newest last*.ckpt, written every max_steps/32 steps; Lightning writes it inside an fsspec transaction
  (lightning/fabric/utilities/cloud_io.py:101-103). The launch exits non-zero, and wq --partial restarts it.
- Not caused by this commit: the handler itself issues `dist.broadcast` from inside the signal handler
  (lightning/pytorch/trainer/connectors/signal_connector.py:116-118), and ranks that see TERM at different step boundaries raise
  at different steps. Both can leave one rank in a collective until the KILL. N changes neither.

b. nice_workers
- `man 2 setpriority`: "under the current Linux/NPTL implementation of POSIX threads, the nice value is a per-thread attribute".
  So one call acts on one thread, and the loop over `/proc/<pid>/task` is needed. EACCES is only for a lower nice value without
  CAP_SYS_NICE; EPERM only when the effective user does not match. Workers are forks of the rank, same user: raising is allowed.
- Threads that a worker starts later inherit the nice value of the thread that creates them (`man 7 sched`: scheduling
  parameters are inherited across fork; clone copies the per-thread value). Worker threads started before the call are covered by
  the task loop.
- Order: `fit_loop.setup_data` calls `iter(self._data_fetcher)` (lightning/pytorch/loops/fit_loop.py:276). With
  persistent_workers the DataLoader creates `_iterator` and starts its workers there (torch/utils/data/dataloader.py:486-488),
  before `on_train_start` (CompileWarmup) and before the first `on_train_batch_start`.
- Attributes: the nice path uses the same chain as the pin path,
  `trainer.datamodule.train_dataset._dataloader._iterator._workers` (host_tuning.py:112-113 and 118-121; `_dataloader` is set at
  gpic_dataset.py:420). The pin path found all workers on every rank in w1048 (d640_pin.log: "[numa] device k: ... 32 workers"
  on all 8 devices). The `is_alive()` filter is kept.
- Autogroup (CONFIG_SCHED_AUTOGROUP=y, enabled on login1): ranks and workers are in one session (Lightning's Popen and the
  DataLoader's fork start no new session; `timeout` only makes a new process group), so the nice value acts between them.

c. Job script
- Both flags come after `--config` in train_ladder.sh (`"$@"` at line 45), so they override lite_ladder.yaml (bucket_cap_mb 1000
  at line 12, train_num_workers 32 at line 253). The same flags were parsed in earlier jobs: `--data.train_num_workers 2`
  (jobs/smoke_ladder.sh:45) and `--trainer.strategy.init_args.bucket_cap_mb` (jobs/smoke_ladder_host.sh:36, w974).
- train_num_workers 16 or 24: `group_window % batch_size` is 1024 % 32 = 0 and does not depend on workers (gpic_dataset.py:407).
  Staggering: chunks = 32, first window of worker i = 32 x max(1, 32(i+1) // n) samples (line 284), a valid size for n = 16, 24.
  Shards: 801 over 8 x 16 = 128 or 8 x 24 = 192 workers, about 6.3 or 4.2 per worker; 2400 steps use 76.8k of about 6.4M samples
  per rank, so no worker reaches the end of its shards. The text cache keeps one fd per worker (lines 247-261); nothing depends
  on the worker count.
- Bucket cap 3000 MiB: trainable parameters are 665 M (d1280_prod.log), fp32 gradients 2.66 GB = 2537 MiB. DDP's first iteration
  uses one bucket (`find_unused_parameters` False → `[sys.maxsize]`, torch/nn/parallel/distributed.py:1173-1174). The rebuild
  after it uses `bucket_bytes_cap` for every bucket and, with a non-default cap, the same value for the first bucket (lines
  1216-1222). 2537 < 3000, so one bucket. The DDPOptimizer split points change, so the subgraphs and their FX cache keys change:
  a cold compile. w974 cold d1280 arms took 423-434 s for 700 steps, so 720 s is enough.
- Memory: with gradient_as_bucket_view the gradients are views of the bucket, so the bucket adds no memory, whatever its size.
  w974 d1280 peaked at 22.8 GB of 80 GB.
- `wait_arm`: started in the background before each arm; the arm runs in the foreground and writes `END <tag> ` before `wait`.
  The loop returns on that line, or after warm-up + 20 s + 15 s of thread_wait + ps. The trailing space keeps d640_w16 and
  d640_w16_nice10 apart. thread_wait with no ranks left prints zero processes and ends. No hang.
- `VAR=1 d640 ...`: bash exports a prefix assignment to the commands run inside the function and drops it after (tested:
  child sees 7, after the call empty). From there env_copy carries it to ranks 1-7. env.sh and ~/.bashrc set no GPIC_ variable.

d. Production with both variables unset: patch_bugs uses 50, as before (same `% 50` test), and HostTuning skips the nice block
(`self._nice` is 0). Production runs from the frozen worktree gpic-wt/ladder_run (0da9b6d), whose patch_bugs.py still has the
fixed `_broadcast_sigterm_every_50`; this commit reaches production only when that worktree moves.

## Not checked
- The autogroup setting on the compute nodes (checked on login1 only).
- Whether 16 workers per rank keep up with d640 at 27 ms per step; the arm answers that.
- Actual compile time of the 3000 MiB graphs; estimated from w974's cold arms.
- The training_step path that could skip an optimizer step on one rank only (it would break DDP before it broke the broadcast).

## Fixes (2026-11-03)

- 1: a new worktree gpic-wt/ladder_host2 at the fixed commit; ladder_kern stays at 89dbf49. The job's WT points to it.
- 2: d1280_control and d1280_b3000 run right after d640_control. --time is 45 min: with finding 3 the expected run is about 30 min.
- 3: both broadcast arms are dropped; a second d640 control at the end gives the spread between launches.
- 4: empty and zero values are handled.
- 5: `[sigterm]` line at import when the variable is set. The logs are checked as listed.
- 6, 7: no change; pt_data_pin's wait in thread_wait.txt is read at analysis.
