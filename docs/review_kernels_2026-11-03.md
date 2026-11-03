VERDICT: GO — no blocker in 9898cc2, and production is unchanged without GPIC_PROF_*; fix S1-S3 first (each is a few lines), and create the worktree before you submit (S0).

Review of 9898cc2 (jobs/smoke_ladder_kernels.sh and the four profiling switches), read against torch 2.7.1 and Lightning 2.6.6 in
envs/pixgen. Nothing was run on a GPU. Line numbers are at 9898cc2.

## Findings

### S0. jobs/smoke_ladder_kernels.sh:21 — the worktree does not exist yet
`$WC/project/gpic-wt/ladder_kern` is missing (`git worktree list` has no ladder_kern). As written, the job exits 2 at line 22
after it has taken a pilot slot.
Fix: `git worktree add --detach $WC/project/gpic-wt/ladder_kern 9898cc2` (or the commit with the fixes below), then submit.

### S1. jobs/smoke_ladder_kernels.sh:105 — TORCHINDUCTOR_MEMORY_PLANNING=1 changes nothing in training graphs; the arm measures a cold compile only
torch/_inductor/codegen/wrapper.py:1227-1230:
`if is_inference and config.memory_planning: self.memory_plan() else: self.memory_plan_reuse()`, with the comment "We disable
planning during training because it presently increases peak memory consumption". compile_fx.py:1954 compiles the forward of a
graph that needs grad with `is_inference=False`, and the backward is never inference. Graphs compiled under no_grad would use
the planner, but the training step has none: the denoiser runs with grad, and the JiT trainer does not call ema_net
(training_JiT.py:36-51 calls only `net`). The only other read of the flag is wrapper.py:1096 (view_as_complex), which this model
does not use. So d640_memplan is a second d640 control with a cold cache. That costs about 8 min of 8 GPUs, and the result can
be misread as "memory planning gives nothing".
Fix: drop the arm. Torch 2.7.1 has no switch that turns the planner on for training graphs.

### S2. jobs/smoke_ladder_kernels.sh:57-79 — the perf reports run inside the profiling window, so the later probes can miss the run
profile_arm runs, in order: sleep 15, perf main 15 s, **two perf reports (up to 2 x 200 s)**, perf autograd 15 s, **two more
reports**, py-spy 15 s, then thread_wait 15 s. That is 75 s of sampling plus up to 800 s of perf report. d640_prod has about
190 s of steps after the warm-up (w974 d640_light: 159 s for 1400 steps, so about 120 s of start-up and 29.6 ms per step; 6000
steps plus 6 last.ckpt saves of 1.6 s each). Reports that need more than about 55 s each (symbol tables of libtorch_cuda.so,
4 reports) push py-spy and thread_wait past the end of training. They then sample nothing or the teardown. Each step is still
written to the log, so the failure is silent. Note that ladder_smoke3_1103_1047_d640_cpp/perf_dso.txt has only 6 samples: that
capture also missed its arm. Also, perf report runs on the same node during the measured steps of d640_prod.
Fix: in perf_thread, record to `$TMPDIR/perf_<tag>_<name>.data` and return. Run the four `perf report` calls after the `wait`
on line 88 (or at the end of the job). The window is then 15+15+15+15+15 = 75 s, and no report runs during the steps.

### S3. jobs/smoke_ladder_kernels.sh:34 — `timeout` without `-k`: one arm that hangs after SIGTERM uses the rest of the job
GNU timeout sends SIGTERM to the process group (python ranks and data workers). Lightning only records the signal and acts at
the next step boundary. A rank killed in the middle of the warm-up or a compile (the inductor compile worker pool also gets
SIGTERM) can leave the other ranks in an NCCL collective. The default timeout for that is 30 min, and timeout never sends
SIGKILL. The arms most likely to reach their limit are the cold-compile arms (the GEMM autotune arm is the least known). w974
had the same pattern, but no arm reached its limit there.
Fix: `timeout -k 60 $limit bash ...`.

### N1. host_tuning.py:90 and gpic_dataset.py:425 — in the cycle path the pin step gets the pids of workers that have ended
`_shutdown_workers` (dataloader.py:1581-1650) joins the workers but leaves `self._workers` filled. `w.pid` still returns the old
pids. pin_rank_threads catches the OSError, but (a) the log line says "32 workers on 44 CPUs" when 0 workers were moved, and
(b) if Linux gives one of those numbers to a new thread or process, `/proc/<n>/task` lists the threads of that process, and
pin_rank_threads moves all of them to `others`. The chance is low (pid numbers seldom wrap in minutes), but not zero.
Fix: `[w.pid for w in getattr(it, "_workers", []) if w.is_alive()]`, or pass `[]` when `getattr(it, "_shutdown", False)`.

### N2. host_tuning.py:62-66 — other threads of the rank stay free to use the reserved CPUs
pin_rank_threads moves only the workers and pt_data_pin to `others`. The rank's NCCL threads (proxy, watchdog), the 7 idle
pt_autograd_<j> threads (j ≠ index), cuda-EvtHandlr and the triton and inductor threads keep the 48-CPU socket mask, so they can
run on the four reserved CPUs. NCCL's proxy thread polls while collectives are in flight. The claim "a CPU that no data worker
uses" is still true, but "no wait for a CPU" is not guaranteed.
Fix: put every thread in `/proc/self/task` except the main thread and pt_autograd_<index> on `others` (this includes
pt_data_pin).

### N3. Verify each switch in the logs before reading results; three switches fail silently
- GPIC_PROF_PIN: any error prints `[numa] pin skipped: ...` and the arm runs as production. Look for 8 lines
  `[numa] device k: main and autograd threads on CPU c`. The thread_wait output of wait_arm (rank 3) must show `allowed 1` for
  `python` and `pt_autograd_3`.
- GPIC_PROF_CYCLE_BATCHES: ignored when group_text_len is false (it is true in lite_ladder.yaml:208). Look for 8 lines
  `[cycle] 64 batches`.
- GPIC_PROF_AGGRESSIVE_FUSION / NO_DDP_SPLIT print nothing. Evidence is a cold compile in the `[warmup] T=` times, and for
  onegraph, fewer compiled graphs in steps_rank0.json → compile.counters (DDPOptimizer splits).

### N4. Header wording, jobs/smoke_ladder_kernels.sh:14-15
"no data path at all" is not exact: each step still does the non_blocking H2D copy from the kept pinned batches (Lightning
moves the batch every step). The arm removes the workers, the pin thread and the shared memory, not the copy. That is the right
bound for the host-side data work. State it that way.

### N5. Expectations for the kernel arms (question d)
- TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1: `use_max_autotune()` (utils.py:1287) is true, so `use_triton_template` lets Triton mm
  templates compete with ATen (backends ATEN,TRITON,CPP; is_big_gpu holds on H100). Epilogue fusion into the chosen template
  also becomes possible. It works with dynamic shapes: for a GEMM whose M contains T, `_is_static_problem` is false, so CUTLASS is
  excluded. Triton and ATen are benchmarked at the **size hint** (the T of the first dynamic compile, 128 in CompileWarmup), and
  that one choice serves every T up to 320. GEMMs with static shapes are tuned exactly. With TORCHINDUCTOR_CPP_WRAPPER=1, the
  autotune runs at compile time (wrapper.py about line 1310 executes a generated autotune block with the hints). Then a chosen
  Triton template launches from the C++ wrapper, and an ATen mm still goes through the shim. Expect: a longer compile on all 8
  ranks at once (the reason for the 1500 s limit), possibly fewer kernels where epilogues fuse, and a different bf16 rounding.
  The host time per step falls only where Triton templates win. It does not turn on CUDA graphs (only mode="max-autotune"
  does). Because the GEMM choice depends on the hint, one arm cannot tell you that Triton is worse at T=320.
- TORCHINDUCTOR_MEMORY_PLANNING=1: see S1. The MemoryPlanner itself handles symbolic sizes (memory_planning.py has
  symbolic_size), and the C++ wrapper has `codegen_alloc_from_pool` (cpp_wrapper_cpu.py:1496). But training graphs never reach
  the planner.
- aggressive_fusion: choices.py:220 then allows fusion of two non-reduction nodes with no shared data, and scheduler.py:3176
  adds candidate pairs grouped by iteration size (horizontal fusion). Expect fewer pointwise kernels and launches (the d640 step
  is host-bound), maybe some register pressure. The change applies at lowering, so it applies when set before the first call.
- optimize_ddp=False: see question c. Expect the first-bucket split to go away. DDPOptimizer uses
  first_bucket_cap = 1 MiB (distributed.py:384-390), so production has at least 2 subgraphs per direction even with 1000 MiB
  buckets. The all-reduce no longer overlaps the backward. With 1000 MiB buckets there is little overlap in production anyway.

### N6. Time budget, --time=01:40:00
The sum of the arm limits is 9000 s, against 6000 s of walltime. The expected total, from w974 and ladder_prof: d640_prod about
330 s, d640_trace 160, d1280_prod 300, d1280_trace 240, 3 x 200 for pin/cycle/cycle_pin, about 470 each for fusion and memplan
(cold d640 warm-up with the C++ wrapper 338 s), about 600 for onegraph, and up to 1500 for gemm. That is about 70-75 min, which
fits. Only the last arm (gemm) is at risk if the earlier arms run long. Dropping memplan (S1) saves about 8 min.

### N7. thread_wait.py — compile workers are not counted
Inductor's compile worker sidecar (`.../compile_worker/__main__.py ...`) does not match "main.py fit", so its CPU is not in
"data workers". This does not matter after the warm-up. Note it if a recompile shows up in TORCH_LOGS.

## Checked and correct

- **pt_autograd_<index>**: libtorch_cpu.so has the "pt_autograd_" prefix. The engine starts one device thread per visible
  device in every process, so each rank has pt_autograd_0..7, and backward work on cuda:k goes to the thread of device k.
  Lightning does not restrict CUDA_VISIBLE_DEVICES per rank, so the index is the local rank. Evidence on this node type:
  w974 run_delay (tmp/ladder_prof/host_prof/ladder_host_1103_1220_d640_light_run_delay.txt) shows `pt_autograd_0` busy
  (43 % CPU) in rank 0, `pt_autograd_1` in rank 1, and so on.
- **The thread exists at the first on_train_batch_start**: device threads start at the first backward. CompileWarmup.on_train_start
  runs four backwards on every rank before step 0, and train_ladder.sh always adds CompileWarmup.
- **os.sched_setaffinity(0, ...)** calls sched_setaffinity(2) with pid 0, which sets the calling thread only. The hook runs on
  the main thread (Lightning calls the hooks there). Calls with a tid set that thread only.
- **Children and threads after the pin**: none in the training path of this job. Checkpoints use TorchCheckpointIO (synchronous
  torch.save on the main thread). limit_val_batches 0 makes evaluation_loop.setup_data return before val_dataloader
  (evaluation_loop.py:164). W&B is disabled. Persistent workers are reused, so there is no re-fork. NCCL communicators and the
  compile sidecar exist since the warm-up (its workers fork inside the sidecar, from its own mask). Only an inductor recompile
  that starts a new pool would inherit the one-CPU mask (TORCH_LOGS=recompiles shows that).
- **`train_dataset._dataloader._iterator._workers` in the normal path**: `_dataloader = loader.pipeline[0]` is the torch
  DataLoader. The persistent iterator is made at the first next() (dataloader.py:486-490), which Lightning's prefetching fetcher
  calls before the first on_train_batch_start. That gives 32 live Process objects, and their threads are listed from /proc.
  Workers forked after setup() carry the socket mask, and their later threads inherit `others`.
- **CPU choice**: w974 logs show devices 0-3 on node 0 and 4-7 on node 1, 48 CPUs each, so `reserved[index % 4]` gives the 8
  ranks 8 different CPUs. The check of 8 ≤ CPUs ≤ 64 catches a rank whose socket binding failed.
- **_CycleBatches in Lightning**: `_check_dataloader_iterable` only calls iter(). `_process_dataloader` leaves objects that are
  not a DataLoader alone. There is no `__len__`, so sized_len gives None and max_batches = inf, the same as the WebLoader.
  CombinedLoader(max_size_cycle) wraps it, max_steps ends the run, and the epoch never ends (the WebLoader epoch does not end
  inside 2400 steps either). `_set_sampler_epoch` finds no sampler and skips it.
- **Repeated batches**: Lightning moves the batch with apply_to_collection, which builds a new tuple, list and dict
  (lightning_utilities apply_func.py, dict fast path), and `.to(device, non_blocking=True)` makes new device tensors.
  `metadata['global_step'] = ...` in training_step changes the copy on the device, not the kept batch. The pinned host tensors
  are only read. All 64 kept batches have T ≤ 320 (collate caps at text_max_length), inside the dynamic range from the warm-up,
  so they cause no recompile.
- **stop_workers in the cycle path**: it joins the pin thread first, then the workers (dataloader.py:1600-1625). The second call
  in main.py:86 does nothing, because `_shutdown` is set. The generator `it` is collected when build_dataloader returns, and
  that is harmless.
- **NO_DDP_SPLIT at compile time**: convert_frame.py:1406 reads `get_optimize_ddp_mode()` for each frame that it compiles. The
  first compile is in CompileWarmup, after configure_model has set the flag on every rank (Lightning's launcher copies the env
  to ranks 1-7). Correctness: in "no_optimization" mode the compiled backward is one autograd node. It returns every gradient,
  the AccumulateGrad hooks fire for each parameter, and DDP's C++ reducer fills its buckets and all-reduces after the compiled
  backward. The result is correct, without overlap. DDP is not static_graph and find_unused_parameters is false.
- **torch._inductor.config / torch._dynamo.config without an import**: torch/__init__.py:2664-2686 imports `_dynamo` and
  `_inductor` lazily in `__getattr__`. torch/_inductor/__init__.py:9 imports `torch._inductor.config`, and _dynamo/__init__.py:13
  imports `config`. Both attributes resolve in configure_model. The FX graph cache key includes the inductor config, so the
  aggressive-fusion, autotune and memory-planning graphs do not reuse or overwrite production entries in the shared
  TORCHINDUCTOR_CACHE_DIR.
- **Environment through `VAR=1 d640 ...`**: bash exports prefix assignments to the commands that a function runs, through
  d640 → arm → timeout → bash train_ladder.sh → python. Lightning copies os.environ to ranks 1-7. w974 used the same pattern
  (`GPIC_PROF_GC_FREEZE=1 d640 ...`) and its arms differed as expected.
- **wait_warm / wait / END matching**: `"END $1 "` has a trailing space, so d640_cycle does not match d640_cycle_pin. A failed or
  timed-out arm writes END, so wait_warm returns. The perf, py-spy and thread_wait calls on a pid that has ended return at once,
  so `wait` cannot hang. `$$` in the background subshell is the job shell's pid, but only one perf_thread runs at a time.
- **rank_pid and thread_wait classification**: /proc/<pid>/environ shows the environment at exec. Rank 0 has no LOCAL_RANK
  (Lightning sets it later with putenv, subprocess_script.py:120). Ranks 1-7 have LOCAL_RANK=k (env_copy, :125). Forked data
  workers show their parent's exec-time block: LOCAL_RANK=k for rank k's workers, none for rank 0's workers. So "parent has a
  different LOCAL_RANK" gives: rank 0 (parent bash, not in the list, -2 ≠ None) → main. Ranks 1-7 (parent rank 0, None ≠ k) →
  main. Workers (parent has the same value, None == None or k == k) → worker. Ranks 1-7 run `python /abs/main.py fit ...` and
  match "main.py fit". The bash and timeout command lines do not match. rank_pid with argument 3 is right. With argument 0 it
  would return nothing, but the job never passes 0. `grep -l .../comm | cut -d/ -f5` gives the tid. thread_wait's % is
  ns / 1e7 / dt. That is correct.
- **perf and py-spy permissions**: perf_event_paranoid = 2 allows sampling your own threads in user space. perf then records
  `cycles:Pu`, as the smoke3 captures show (12K and 20K samples), so kernel time (futex, scheduler) is not in the dso tables.
  py-spy wrote pyspy_raw.txt in the smoke3 runs.
- **BENCH_PROFILE windows against saves**: max_steps is 32000 in the arguments, so last.ckpt is saved every 1000 steps and the ms
  files every 8000 steps. 1205:1208 (stop 1300) and 505:508 (stop 600) contain no save. LADDER_PARTIAL_OK stops the rc 99 check.
  A separate LADDER_RUN_ROOT per tag means no arm resumes from another arm.
- Production: with no GPIC_PROF_* set, `_pin` is False, n is 0 and both compile flags stay at their defaults. The only new code
  that runs is one `os.environ.get` per process and one `if` per step.

## Not checked

- The C++ source of the autograd engine (only the strings in the .so and the w974 per-thread data). The count of 8 device
  threads per rank is inferred, not read.
- Whether the H100 nodes have SMT. If they do, `socket[-4:]` can be the hyperthread siblings of CPUs in `others`. The 96 CPUs
  on 2 x 48 suggest no SMT. Check on the node:
  `cat /sys/devices/system/cpu/cpu47/topology/thread_siblings_list`.
- How long `perf report` takes on these libraries (S2 gives the range that matters).
- Whether max-autotune GEMM with the compile-time autotune of the C++ wrapper compiles without error for these backed dynamic
  shapes. I found no rejection in the code. If it fails, the arm's rc and log show it.
- The compile time of one graph without DDPOptimizer, and of the GEMM autotune, against their 900 s and 1500 s limits.

## Fixes (2026-11-03)

- S0: the worktree ladder_kern is created at the fixed commit before the submit.
- S1: checked in torch/_inductor/codegen/wrapper.py:1228. The arm d640_memplan is dropped; the job header says why.
- S2: profile_arm only records (75 s). perf_reports runs after the arm ends, and the .data files stay in the profile directory.
- S3: `timeout -k 60`.
- N1: only live workers go to pin_rank_threads; the log line counts the workers really moved.
- N2: every thread of the rank except the main thread and pt_autograd_<index> goes to the other CPUs.
- N3: read at analysis: 8 `[numa] device k: main and autograd threads` lines, `allowed 1`, 8 `[cycle]` lines, cold compile times.
- N4: the job header says what the cycle arm removes and what it keeps (the copy to the GPU).
- Also: `pgrep -u $(id -u)` (a fetched job has a clean environment); thread_wait.py prints system-mode CPU and minor faults; NCCL_DEBUG=INFO in the two trace arms.
- Not changed: N5 (expectations), N6 (budget: about 65 min without memplan; --time stays 1:40), N7 (no matter after the warm-up).
- SMT: not present. Xeon Platinum 8468, 48 cores per socket and 96 CPUs per node, measured earlier in this work.
