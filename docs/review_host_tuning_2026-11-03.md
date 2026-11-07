VERDICT: GO for the diff (a5ab18d) after S1; bucket_cap_mb=1000 as planned does NOT do what it claims (S2, S3): fix or A/B it first.

# Review: host tuning, 2026-11-03 (git diff 6d57702 HEAD, plus planned bucket_cap_mb=1000)

### S1 (should-fix) baselines/PixelGen/src/callbacks/host_tuning.py:35-36: a binding failure kills the run
Any error in bind_to_gpu_socket raises in setup on every launch: nvidia-smi missing from PATH (a guest runs under `env -i`, fetch_pending.sh:294), a uuid KeyError, `numa_node` = -1 (then node-1/cpulist does not exist), a sysfs read error. The run then fails at start on every relaunch on that node, for an optimisation worth a few percent.
Fix: wrap the call in `try/except Exception as e: print(f"[numa] skipped: {e!r}")`, and skip when node < 0.

### S2 (should-fix) planned config bucket_cap_mb: 1000: one bucket only up to 262M parameters
bucket_cap_mb is MiB (torch/nn/parallel/distributed.py:809, ×1024×1024). Gradients are fp32 (bf16-mixed keeps fp32 parameters), so 1000 MiB holds 262M parameters. The sizes above that (up to 665M parameters, 2.66 GB) get 2 to 3 buckets, not one.
Fix: if one bucket is the goal, set the cap from the largest size: 2600 MiB or more (3000 is safe). The count is 665M elements, under 2^31, so no NCCL count limit applies.

### S3 (should-fix) torch/_dynamo/backends/distributed.py:386-388,441-444 and convert_frame.py:1412: DDPOptimizer still splits the graph
convert_frame passes only bucket_bytes_cap. DDPOptimizer therefore keeps first_bucket_cap = _DEFAULT_FIRST_BUCKET_BYTES = 1 MiB (checked on vis2: 1048576), and it cuts a new graph after the first 1 MiB of parameters in reverse order, whatever bucket_cap_mb is. Result: each compiled module with gradients still has at least 2 graphs; the claim "no DDPOptimizer split" is false. DDP itself uses first bucket = cap when the cap is explicit (distributed.py:1216-1221), so DDP's buckets and the graph split also stop matching.
Fix: to get one graph and one all-reduce, set `torch._dynamo.config.optimize_ddp = False` before the first compile (DDP then fires its hooks after the one compiled backward), and keep a large cap. Otherwise state the claim as "fewer graphs".

### S4 (should-fix) one bucket removes all overlap of the all-reduce with the backward
With one bucket, the all-reduce starts only after the whole backward. Estimate (about 350 GB/s bus bandwidth, ring 2(n-1)/n): d640 0.44 GB is about 2.2 ms exposed on a 32.6 ms step (~7%); d1280 2.66 GB is about 13 ms exposed. The 25 MiB default hid most of this. jobs/smoke_ladder_host.sh has no bucket arm, so the gain is not measured.
Fix: A/B the step time at d640 and d1280 (bucket default vs cap 3000 with optimize_ddp False) before production.

### N1 jobs/train_ladder.sh:43: LD_PRELOAD overwrites, and the large-alloc report
Use `export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4${LD_PRELOAD:+:$LD_PRELOAD}`. gperftools prints "tcmalloc: large alloc" for every malloc > 1 GiB (none in the smoke logs); set TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=1e11 so a long run cannot fill the log. gperftools does not return freed memory to the OS quickly: watch the RSS of rank 0 over a whole launch (800G is ample for 8 ranks + 256 workers).

### N2 baselines/PixelGen/src/callbacks/host_tuning.py:29: each socket now carries 4 ranks × (32 data workers + 32 Inductor compile workers, compile_threads=32)
Same total load as before, only without migration between sockets. The compile pool forks at the first compile (after setup), so warm-up compile runs on 48 CPUs per 4 ranks. Not a correctness issue; the warm-up time printed by CompileWarmup shows it.

### N3 jobs/smoke_ladder_host.sh:36,42-46: the GPIC_PROF_{NUMA,GC_FREEZE,NO_SIGTERM_BCAST,LOG_ONLY_WHEN_WRITTEN} switches are gone
The script still runs, but every arm now gets HostTuning, tcmalloc and log-only-when-written from train_ladder.sh, so the arms compare nothing (only NO_SIGTERM_BCAST is no longer available at all). No other job or tool references the four names. Mark the script as historical or rewrite the arms as on/off of the new defaults.

## Checked and correct
- Order (lightning trainer.py:1033-1079, fit_loop.py:208-276,414-433): setup_environment (process group) → callback setup → restore → strategy.setup (DDP wrap) → W&B start (1062) → fit_loop.setup_data, where `iter(data_fetcher)` forks the 32 persistent workers and starts the pin thread → on_train_start. So the binding precedes the workers, the pin thread, the autograd thread (first backward) and W&B, which all inherit it. NCCL threads made at init keep all CPUs (not starved).
- root_device.index is the local logical index; get_device_properties(index).uuid lookup does not depend on index order. torch gives the bare uuid, nvidia-smi "GPU-<uuid>"; the last smoke shows correct lines ("device 5 0000:bb:00.0 node 1: 48 CPUs"). The `[4:]` slice is right for nvidia-smi's 8-digit domain; `.lower()` matches sysfs. Empty intersection is handled (line 28).
- Callback order: _reorder_callbacks moves only ModelCheckpoints to the end; HostTuning stays after CompileWarmup; BenchTimer after HostTuning. gc.freeze runs after warm-up and after the checkpoint restore; later objects are collected normally; workers were forked earlier, so freeze does not touch them.
- log_dict: nothing in src reads callback_metrics/logged_metrics; both monitors are null; progress bar off; ModelCheckpoint's `{step}` comes from _monitor_candidates, not logged values. should_update_logs uses _batches_that_stepped + 1 both inside training_step and in update_train_step_metrics (training_epoch_loop.py:384 before the increment at 415), the same test train/lr already used, so the same steps write the same values. on_step only, so no epoch reduction to break. Lightning does not warn on intermittent logging.
- tcmalloc: gperftools-libs 2.9.1 present on vis2 and nodes; torch + 3 GB tensor + fork Pool under LD_PRELOAD ran fine on vis2. A missing or unloadable file: the `[ -f ]` guard, and ld.so ignores an unloadable preload with a warning. It is exported after gpu_health.py.
- Bucket math: DDP divides by world size and sums the same gradients; only NCCL's sum order changes with message size (fp32 rounding). gradient_as_bucket_view stays true for any cap. find_unused_parameters False → the first iteration already uses one bucket (distributed.py:1173-1174); rebuild after it. Bucket layout is not in the checkpoint: resume across cap settings works (Inductor recompiles once). DDPStrategy passes bucket_cap_mb to DDP like broadcast_buffers. The CompileWarmup comment stays true: the warm-up compiles under _inside_ddp_forward, with the new cap.

## Not tested
- No GPU run: S4's numbers are estimates; DDPOptimizer's real split count at each size; tcmalloc RSS over hours; behaviour on a node where numa_node is -1 (none known on h100 pli).

## Fixes after this review
- S1: a binding failure prints a line and the run goes on. N1: LD_PRELOAD keeps what was set; the large-alloc report is off. N3: the smoke script says it is historical.
- S2, S3, S4: jobs/smoke_ladder_host.sh does have bucket arms (100 and 1000 MiB). With 1000 MiB d640 gets one DDP bucket and d1280 three, and DDPOptimizer's first cut stays, so the claim is "fewer graphs and fewer all-reduces". The cap for production follows the measured arms at d640 and d1280.

