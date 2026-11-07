VERDICT: NO-GO — the worktree gpic-wt/ladder_host3 does not exist, so the job exits 2 at line 16; add it (finding 1), then GO.

Review of 6219039 (GPIC_WORKER_IDLE in src/callbacks/host_tuning.py, jobs/smoke_ladder_host3.sh). Nothing ran on a GPU. Nothing was
submitted. One CPU test ran on vis2. Package paths are relative to envs/pixgen/lib/python3.11/site-packages.

## Findings

### 1. jobs/smoke_ladder_host3.sh:15-16: the worktree does not exist (blocker)
`ls ../ladder_host3` fails, and `git worktree list` has no ladder_host3 entry (ladder_host2 is at de7eae1, ladder_seed at 1882658).
Failure: the job prints "no worktree" and exits 2 within seconds. A host spends a pilot slot on it, and the job produces nothing.
Fix: `git worktree add --detach /scratch/gpfs/GROUP/USER/project/gpic-wt/ladder_host3 6219039` before the submit (or the
commit that carries the fixes below).

### 2. The scheduler model in the docstring is right, but the gain depends on the CPUs staying unsaturated. Watch the late-batch stalls.
- Preemption at wakeup: the running kernel has the 6.8-era scheduler names `check_preempt_wakeup_fair` and `wakeup_preempt` in
  /proc/kallsyms, and no `check_preempt_wakeup`. So RHEL 9.8 back-ported the upstream scheduler code, and "5.14 CFS" is only
  roughly true. I could not tell from the symbols whether EEVDF is in (there is no `pick_eevdf` symbol, but it can be inlined). The
  rule is the same either way. Upstream, in both CFS and EEVDF, the wakeup check does `find_matching_se`, then
  `if (cse_is_idle && !pse_is_idle) goto preempt;`. A waking SCHED_NORMAL thread (nice 19) preempts a running SCHED_IDLE task at
  once. This holds when both are in the same group. Ranks and workers share one autogroup (CONFIG_SCHED_AUTOGROUP=y,
  sched_autogroup_enabled=1 on login1 and vis2; review_host2 b), so the match is at the task level.
- Placement: `select_idle_sibling` and `select_idle_cpu` accept `available_idle_cpu(cpu) || sched_idle_cpu(cpu)` (since 5.4/5.6).
  `sched_idle_cpu` is true when `rq->nr_running == rq->cfs.idle_h_nr_running`, that is, when the CPU has only SCHED_IDLE tasks
  runnable. So a CPU that runs only workers counts as idle for a waking rank thread.
- Weights: SCHED_IDLE is WEIGHT_IDLEPRIO = 3, and nice 19 is 15. SCHED_IDLE is not strict: when a worker shares a CPU with a
  normal thread that does not sleep, the worker still gets 3/18 = 17% of that CPU. So a worker cannot starve forever, and it can
  starve only on CPUs that normal threads keep busy.
- Saturation is plausible at d640. docs/text_cache.md:45 says the workers use about 70 of 96 CPUs at 30 ms per step. The rank
  threads add main, autograd, pin and NCCL/CUDA threads: about 16-30 CPUs on 8 ranks. That is close to 96. If the step gets
  faster, the workers need more CPU. When normal threads fill a socket, the workers get about 1/6 of the contested CPUs. The data
  path then binds, and a rank waits for its batch. This is the same kind of stall as the w1048 pin arm (text_cache.md:51: p99
  164.6 ms, steps of 0.3 s or more, workers' CPU wait 37 CPU-equivalents). In-order delivery (dataloader.py:258, `in_order=True`;
  WebLoader does not change it) means one late worker stalls its rank, and DDP then stalls all 8 ranks. The buffer is 32 workers x
  prefetch 4: worker i serves 1 batch in 32, about 0.9 s apart at 27 ms per step, so a worker must fall behind by about 3 s to
  stall the step. A short preemption does not stall it.
- Priority inversion, by holder:
  - resource_sharer thread (review_host2 finding 7): it is now SCHED_IDLE. The pin thread (normal) blocks on its socket until it
    runs. On an idle or SCHED_IDLE-only CPU it runs at once. On a CPU busy with a normal thread it can wait up to that thread's
    slice (a few ms). It waits longer only under saturation. This is the most direct inversion on the pin path.
  - multiprocessing.Queue: the result queue's `_wlock` is shared by the 32 writers (the workers' feeder threads). The reader lock
    `_rlock` is the pin thread's alone. A preempted feeder that holds `_wlock` blocks only the other workers' feeders. The pin
    thread sees it only as a late batch. The messages are small (fds and metadata), so the lock is held for microseconds.
  - Kernel locks: the kernel has no priority inheritance for SCHED_IDLE holders of mutexes or rwsems (only rt_mutex does PI, for
    RT). mmap_lock is per mm, and each worker has its own mm, so a holder blocks only its own worker. GPFS reads of shards and the
    shared text-cache file are reads, so the inode locks are shared. Rank threads do not read GPFS in steady state; checkpoint
    writes happen every max_steps/32 steps, on other files. shm pages: the rank maps /dev/shm files that the worker wrote and then
    released. The only shared state is the page lock during a fault, which is held briefly. Risk: low while CPUs are free.
- Where it shows: BenchTimer's per-step times (steps_rank<r>.json): p99, and the count and sum of steps of 0.3 s or more, against
  d640_control/control2. BenchTimer's data wait on rank 0. thread_wait.txt: the wait of pt_data_pin, and the workers' summed CPU
  wait. A rise in the workers' wait is expected and harmless, as long as the stalls do not rise.
Fix: none needed before the job. Do not adopt the switch if p99 or the 0.3 s stall count rises against both controls, even when
the median falls.

### 3. jobs/smoke_ladder_host3.sh:4, 46-56: the arm limits sum to 42 min against --time of 35 min
Limits: 6 x 420 s = 2520 s. Expected, from w1051 (ladder_host2_1103_2348/phases.log): d640 2400 steps 184-195 s, d1280 800 steps
243 s. Four d640 arms and two d1280 arms take about 13 + 8 = 21 min, plus the health gates. So the expected run fits. A d640 arm
that hangs uses 420 s, about twice its normal time. If one hangs, the last arm, d1280_idle_b200, can hit the job's --time.
Fix (cheap): d640 limit 300 s (1.6 times 190 s). Then 4 x 300 + 2 x 420 = 2040 s = 34 min fits under 35 min. Or set
--time=00:45:00.

### 4. host_tuning.py:83-88: a thread that ends mid-loop leaves the rest of that worker normal (minor)
`try` wraps the whole thread loop of a worker. ESRCH on one thread (it ended after `listdir`) skips that worker's remaining threads,
and the worker is not counted. The log then shows "31 of 32", while part of that worker still runs normal. Effect: small. Most
threads run under SCHED_IDLE, and the log does show the miss.
Fix: put the try around each `sched_setscheduler` call, and count a worker when its main thread (tid == pid) succeeded.

## Checked and found correct

- Inheritance, tested on vis2 (tmp/idle_t.py: a forked worker with one thread started before the call and one after, then a
  fork): all four threads and the grandchild report policy 5 (SCHED_IDLE). The code passes no SCHED_RESET_ON_FORK flag. Nothing in
  src/ calls sched_setscheduler, setpriority or os.nice besides this function. The worker cannot switch back:
  `sched_setscheduler(0, SCHED_OTHER)` gave EPERM. RLIMIT_NICE is 0 (`ulimit -e` = 0), and sched(7) requires the nice value to fit
  RLIMIT_NICE to leave SCHED_IDLE. Moving one's own children to SCHED_IDLE needs no privilege: the call succeeded at nice 19.
- Call path: the same chain as the pin path (`train_dataset._dataloader._iterator._workers`, gpic_dataset.py:420). w1051 found the
  workers on all devices ("0 of 32", "0 of 16": the workers were found, and setpriority to a lower nice failed with EACCES, as
  expected at nice 19). The flag clears after the first batch. Workers persist (gpic_dataset.py:416), so they keep the policy
  across epochs. A --partial resume is a new process, and its first batch applies the switch again. Workers are forked by the main
  thread (normal), so no new process starts as SCHED_IDLE by accident. GPIC_PROF_CYCLE_BATCHES, which stops the workers, is not
  set in this job.
- `ps -u $(id -u) -L -o cls=,ni=,comm=` is valid. On vis2 it prints `IDL   - python`: the class is IDL and NI is "-" for
  SCHED_IDLE. Expected in policy.txt: IDL lines for pt_data_worker in the idle arms; TS 19 for the rank threads.
- Prefix assignments on a function with two variables: `A=1 B=200 f` passes both to the child bash, and both are empty after the
  call (tested). From there train_ladder.sh passes them to main.py, and Lightning's env_copy passes them to ranks 1-7.
  patch_bugs.py:31-32 prints `[sigterm] broadcast every 200 steps` on each rank.
- Arms do not disturb each other: they run in sequence. `wait` joins each wait_arm before the next arm. Each arm has its own
  BENCH_OUT and LADDER_RUN_ROOT. Each launch reads other shards (GPIC_SHARD_SEED is the launch time), and control2 gives the spread
  between launches. d640_idle_b200 and d1280_idle_b200 change two things. The b200 part is arithmetic: (1/50 - 1/200) x 30 ms =
  0.45 ms at d640 and 0.83 ms at d1280 (review_host2 finding 3). Subtract it, or read d640_idle against d640_idle_b200.
- Math: SCHED_IDLE changes only when threads run. Delivery is in order (`in_order=True`), the worker shard split does not depend on
  timing, and the worker count is unchanged. So the batches and their order are the same for a given GPIC_SHARD_SEED. Same math.
- Production with both variables unset: `self._idle` is False, and the broadcast stays every 50 steps (patch_bugs.py:30). Production
  also runs from gpic-wt/ladder_run at 0da9b6d (train_ladder.sh:22), which has neither switch.
- Adoption without a resubmit: wq snapshots the outer script, train_ladder.sh, and that script runs main.py from the worktree
  `${LADDER_WT:-ladder_run}`. A new env export in train_ladder.sh would not reach jobs that are already queued. A code default
  does: host_tuning.py `os.environ.get("GPIC_WORKER_IDLE", "1") != "0"` and patch_bugs.py `or 200`, with ladder_run moved to that
  commit. Every later launch and every --partial resume imports the new code. Running launches keep the code they imported. Since
  0da9b6d, the other changes to src/ and train_ladder.sh are env-gated profiling switches and the GPIC_SHARD_SEED override, so
  moving ladder_run brings nothing else. KILL comes 45 s after TERM. At 200 steps one broadcast interval is 5.4 s (d640) to 17.6 s
  (d1280, the largest size in ladder_runs.json). Ranks raise on SIGTERM at the end of their step anyway (review_host2 a).
- eval_ladder.sh: predict uses lite_ladder_sample.yaml, whose callbacks are CheckpointHook and SaveImagesHook. HostTuning is added
  only by train_ladder.sh's `--trainer.callbacks`. Predict has no on_train_batch_start, and the broadcast patch sits on the
  training epoch loop. A code default is safe for evaluation.

## Not checked

- Whether this RHEL 9.8 kernel has EEVDF or CFS (no debugfs access; no kernel source on the node). The two rules used above,
  idle-vs-normal wakeup preemption and sched_idle_cpu in placement, are in both.
- The autogroup setting on the compute node (checked on login1 and vis2 only). With autogroup off, the result is the same: both are
  then in the job's cgroup.
- Whether the CPUs saturate at d640 under the switch. The arm measures it (finding 2).

## Fixes (2026-11-04)

- 1: the worktree gpic-wt/ladder_host3 is created at the fixed commit.
- 2: read at analysis: the switch is not adopted if p99 or the count of steps of 0.3 s or more rises against both controls.
- 3: d640 arm limit 300 s; the sum is 34 min against --time 35 min.
- 4: the try is per thread, and a worker counts when its main thread moved.
