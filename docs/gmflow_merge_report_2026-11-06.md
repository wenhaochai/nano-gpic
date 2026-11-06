# GMFlow merge report (2026-11-06): gmflow-hn and gmflow-b2 into gmflow, and the review fixes

Branch `gmflow`, worktree `gpic-wt/gmflow`. Base: `4ca9098` (group D code plus its review). Nothing was pushed. No job was submitted or cancelled. No other worktree was touched. All Python ran on vis1 / vis2 (CPU). The login node only edited files and ran git.

## Result

1. Both merges are committed: `48258ef` (gmflow-hn at `3c08a5a`) and `4545a6f` (gmflow-b2 at `a7a28f6`).
2. Every BLOCKING, HIGH and MEDIUM finding of both reviews is fixed, except the group D findings that concern only the second-order samplers (H1, H2, M1). The owner decided on 2026-11-06 not to run GM-SDE 2 / GM-ODE 2; their code and tests stay as they are. The cheap LOW findings are fixed too (list below).
3. The full suite passes on vis1: T1–T14 and T21–T30, 24 tests, `ALL PASS`, exit 0.
4. Mutants: 17 of 17 killed (the 10 of group D's branch, the 4 of gmflow-hn, 3 new for the merge's guards).
5. `bash -n` passes on every changed job script.
6. The frozen worktree for the GPU jobs is `gpic-wt/gmflow_run3`, to be created by the owner at the final commit of this series. A checkout of this commit passes every content guard of the job scripts (checked on this worktree's files). T27 and T30 run the scripts against a stub `gmflow_run3`.
7. Found and fixed on the way: gmflow-hn's `tools/gmflow/bench_sampling.py` did not compile (`dict(per_call_fp32_max_lt_1e-4=...)` is a SyntaxError). No CPU test imported it, so the N benchmark job would have died at start (fixed in `97390f4`).

## Commits (oldest first)

| Commit | What | Closes |
|---|---|---|
| `48258ef` | Merge gmflow-hn. `sampling_gmflow.py:97-108`: gmflow's docstring and `order`, hn's `spread_table` and `fused`; loop order point mixture → order 2 → fused or unfused sub-steps (`:117-132`); `assert not (spread and order 2)` (`:106`). `eval_gmflow.sh`: gmflow's `EVAL_SPREAD`, `SPREAD_ARGS` and offset-head guard plus hn's `gm_ode_cal`, `EVAL_SPREAD_TABLE`, `EVAL_SUBSTEPS` (now also `gm_ode2`); `gm_ode_cal` in the usage line. `evals_gmflow.sh`: `--link`, `--wt`, `--spread`, `--table`; 4-field tasks; `EVALS_WT` dropped. `mutate.py`, `test_gmflow.py`: both blocks; T27 on `--wt` with a stub worktree under a temporary `$WC`. `docs/gmflow_hn*.md`: `EVALS_WT` → `--wt`. | hn F6, F7 (resolution), D H3(b), D M2 |
| `4545a6f` | Merge gmflow-b2: `jobs/continue_run.sh`, `jobs/train_seed.sh`, `docs/gmflow.md` section 8 (sections 7 and 8 kept in number order). | — |
| `5e313f0` | `test_gmflow.py:218, :605`: `torch.cdist(..., compute_mode="donot_use_mm_for_euclid_dist")`. | D H3(a) |
| `e1e7a6a` | `sampling_gmflow.py:145-154` `_substep_sigmas`: the fused path's σ grid in three device ops from cached fp32 weights `j / n`; bitwise equal to the loop's grid (T25 asserts it for n ∈ {3, 7, 16, 32, 64} × calls ∈ {50, 8, 4, 2}). `bench_sampling.py:59`: `set_device` before `init_process_group`. | hn F11 |
| `17ba320` | `gm_spread.py:38-51` `check_bound`: at every call time t > 0 the sampler reads, s_x(t) ≤ (1 − t)/t, compared in fp64, else `ValueError` with every offending call. `:54-59` `check_table_ckpt`: basename(table["ckpt"]) must name ms-step=<step>. `:73-90`: `python -m src.diffusion.flow_matching.gm_spread check <table> <step> <calls>` (exit 2 on refusal). `sampling_gmflow.py:107-108`: `GMODESampler(spread_table=...)` calls `check_bound(self.timesteps[:-1])`. | hn F1 (ckpt part), F2/F3 per the owner's decision |
| `3d0ff80` | `eval_gmflow.sh`: `GMFLOW_WT` required (`:31`); one `eval_dir` rule (`:33-36`, the same text in `evals_gmflow.sh:36-39`): `gm_ode_cal` → `eval_<step>_gm_ode_cal-<sha1[:8] of the table file>_calls<calls>[_n<n>]`; calls × sub-steps > 512 refused for gm_ode / gm_ode2 / gm_ode_cal (`:52-54`); table `realpath` (`:66`); the pre-check `gm_spread check` before the prediction (`:85-87`); `ckpt.txt` records the table's full sha1 (`:104`); header: submit through `evals_gmflow.sh` (`:24-25`). `evals_gmflow.sh`: `--wt` takes only `[A-Za-z0-9_-]+` (`:26`); `--table` realpath (`:30`); a `gm_ode_cal` task without `--table` is refused (`:44`); done-check by `eval_dir` (`:58-61`). `smoke_gmflow.sh:22` sets `GMFLOW_WT`. | hn F1, F5, F8; D M3, M4, L5 |
| `27b2f9e` | Tests: T11 alignment (offset bias r·16 + c lands where the means' channel lands, K = 8 and 1); T26 fused on the per-pixel logstd of `JiT_T2I(gm_spread="pixel")` at 4 × 8 and 8 × 16; T27 rewritten (hashed name, two tables, relative path, 512 limit, `GMFLOW_WT`, stub `gmflow_run3`, `eval_dir` identical in both scripts); T28 (bound guard, ckpt check, the CLI pre-check, calls × n = 1024). | D L1; tests asked by both reviews |
| `9e0cb7a` | `calibrate_spread.py:55-58, :71`: `noise_for(index, n_t, ...)` with `len(ts)`; `:130`: merge asserts equal `images`. T24 without the `T_GRID` patch, plus a mismatched-`images` refusal. `calibrate_spread.sh:17, :35`: `WT=gmflow_run3`; `commit.txt` in each `<step>/`. `bench_sampling.sh:18`: `WT=gmflow_run3`. | hn F9, F10, P |
| `97390f4` | `bench_sampling.py:146-176`: on an uncompiled fp32 copy of the K = 8 net, autocast and TF32 off, each rank's first 32 prompts: end to end fused vs not fused at 8 and 4 calls, and the per-call check at the last call (t = 1 − 1/calls, σ 1/calls → 0); `:187-194`: both gathered over ranks, pass if max |diff| < 1e-4; the bf16 numbers are kept as `end_to_end_bf16_report_only`. The SyntaxError fix. | hn F4 |
| `2c0de5f` | `train_gmflow.sh:32-37, :47, :63-66, :71`: `--lam <value>` after `--spread` (default 0.5; only `0.x` or `1`); passes `--model.diffusion_trainer.init_args.lam`; `lam.txt` (a run with a checkpoint and no `lam.txt` is 0.5); another value refused; `--wt` name check (`:30`). T29 and T30. | owner's popup; D L5 |
| `8976ca6` | `smoke_gmflow2.sh`: no gm_ode2 / gm_sde2 and no paired diff; pixel model gm_ode at 8 and 4 calls; the guard counts only with its own assert text (`:73-77`); w1173 gm_ode at 8 calls (`:81-84`); `WT=gmflow_run3` (`:21`); guard on `check_bound`. Worst case 480 + 300 + 2·240 + 180 + 240 s = 28 min under the 50 min wall. | owner's decision; D L3, L6 |
| `72adb92` | T26: fused = not fused with order 2 on the per-pixel model. `mutate.py`: three mutants for the merge's guards. | hn F7 (test) |
| `0689638`, `d36fbf8` | `docs/gmflow.md` 7.2: H1 (clamp: order 2 = order 1 at ≥ 32 calls) and H2 (the official clamp moves the mean by up to about 4/255 per call at var ~ 1e-5) as notes; the owner's decision on order 2. 7.3: the pixel runs' ms points (15552 / 16608; C0/16 = 31104 / 33216). `gmflow_hn_impl_report` §5 item 1: the scaled-table follow-up withdrawn. `gmflow_hn.md`: the hashed name and the guard. | D L2; hn F3 (docs); H1/H2 docs only |
| `f2fc0aa` | `docs/gmflow.md` sections 9, 10, 11 (Chinese). | — |
| `0b3a6d0` | `mutate.py`: `table_wrong_direction` pattern follows the fp64 refactor of `SpreadTable`. | — |
| this file | the report | — |

### Findings not fixed

| Finding | Why |
|---|---|
| D H1 (order 2 is order 1 at ≥ 32 calls) | Second order only. No job runs it (owner 2026-11-06). Documented in `docs/gmflow.md` 7.2 item 8. |
| D H2 (no 1e-6 clamp on var − v′; up to about 4/255 per call) | Second order only. Code kept as is; documented in 7.2 item 6. |
| D M1 (the smoke tests order 2 only at 8 calls) | Second order only. The smoke no longer runs order 2. |
| D L4 (recompiles after warm-up are counted, not failed) | Left as a number to read in the smoke log. A fail on > 0 would also fire on the known Qwen3 recompile (batch 32 → 1) in predict. |
| hn F5, root cause (gm_convert's clamp `norm ≥ 1e-6`) | Kept, so finished evaluations keep their numbers (owner's instruction). The eval refuses calls × n > 512 instead; T28 documents the clamp: rms to the exact step 1.5e-7 at 8 × 64 and 7.1e-3 at 8 × 128. |

## Tests (vis1, Xeon 8380, CPU; full log `tmp/merge_0611/tests_vis1.txt`)

Run: `ssh vis1 'bash .../tools/gmflow/run_vis.sh tools/gmflow/test_gmflow.py'` on a clone of `72adb92` (the later commits change only docs and `mutate.py`).

| Test | What | Result |
|---|---|---|
| T1 | K = 1, λ = 1, fixed s_u: gradient on μ = v-loss gradient × C/c² | PASS |
| T2 | NLL over x vs over v differ by a constant | PASS |
| T3 | c1, c2, c3 by Monte Carlo; λ = 1 gives (0, 1, 0) | PASS |
| T4 | transition loss minimized at the truth | PASS |
| T5 | GM-SDE with the exact posterior on 2-point data | PASS |
| T6 | gm_convert vs grid Bayes | PASS |
| T7 | shapes and modes | PASS |
| T8 | init, finite at t near 0 and 1 | PASS |
| T9 | FLOPs = hand count; STEPS | PASS |
| T10 | 1 sub-step = DDIM on the mean; final_mean | PASS |
| T11 | image = c16d71d bitwise; pixel at init = image; detach; **new: offset at (r, c) = means' (r, c)** | PASS |
| T12 | per-pixel posterior std recovered by the transition loss | PASS |
| T13 | per-pixel spread in samplers and loss vs grid Bayes | PASS |
| T14 | second order vs the transcription of the official code | PASS |
| T21 | table interpolation | PASS |
| T22 | s_x = 0 table = no table bitwise = exact step (2.4e-7 – 4.2e-7) | PASS |
| T23 | Gaussian toy: true-std table beats the exact step 15–52× | PASS |
| T24 | calibration recovers the posterior std; sharding; merge refuses a missing shard **and mismatched images** | PASS |
| T25 | reference = sampler loop; **fused σ grid = loop grid bitwise** | PASS |
| T26 | kernel = reference (TRITON_INTERPRET); fused = not fused; **per-pixel JiT_T2I: 6.0e-7 (4 × 8), 3.8e-6 (8 × 16); order 2: 3.3e-7** | PASS |
| T27 | eval / evals names and arguments: hashed `gm_ode_cal` name (two tables → two directories), relative table path, 512 limit (8 × 64 accepted; 8 × 128, 50 × 11, 4 × 129 refused), `GMFLOW_WT` required, `--wt gmflow_run3` on a stub, `../x` refused, done-check by table hash, `eval_dir` identical | PASS |
| T28 | bound guard (smoke-like table accepted at 8 and 4 calls, refused at 16 and 50; at the bound passes, 0.1% above fails; order 2 + table refused); ckpt check (066816 accepted for 66816, refused for 33408, 6681); CLI pre-check exit 0 / 2; clamp at calls × n = 1024 | PASS |
| T29 | λ = 1: loss = 2 × plain NLL of x under the x-mixture (values 1e-6, gradients 1e-5), K 8 and 1, image and pixel; real model finite at t = σ(−12), 0.5, σ(12) | PASS |
| T30 | `train_gmflow.sh`: old call unchanged (lam.txt 0.5); `--wt gmflow_run3 [--spread pixel] --lam 1` passes `init_args.lam 1`; restart with another λ refused; old run without lam.txt = 0.5; bad values refused | PASS |

Also checked:
- `main.py fit --config lite_ladder.yaml --config gmflow_train.yaml --model.diffusion_trainer.init_args.lam 1 --model.denoiser.init_args.gm_spread pixel --print_config` resolves `lam: 1`, `gm_spread: pixel`, `gm_k: 8` (vis1).
- The w1147 CPU smoke table (8 images, `tmp/vis_hn/cal_smoke/table.json`): `gm_spread check` accepts it at 8, 4 and 16 calls and refuses it at 50 calls (t = 0.96: s_x 0.0562 > 0.0417; t = 0.98: 0.0549 > 0.0204). The real table (4096 images) is checked by the eval itself.
- `py_compile` of every file in `tools/gmflow/`.
- `bash -n`: `bench_sampling.sh`, `calibrate_spread.sh`, `continue_run.sh`, `eval_gmflow.sh`, `evals_gmflow.sh`, `smoke_gmflow.sh`, `smoke_gmflow2.sh`, `train_gmflow.sh`, `train_seed.sh`: all OK.
- The merged commit `4545a6f` alone: T1–T14, T21–T27 ALL PASS on vis1 (before any fix).

## Mutants

Each mutant goes into a temporary copy of `baselines/PixelGen/src`, and the full suite runs on the copy. The 17 were split over two nodes to save time: vis2 ran mutants 1–8 (`tmp/merge_0611/mutants_vis2.txt`; that run was then stopped, because the rest of its list repeated mutants already done on vis1), vis1 ran 9–17 (`mutants_vis1.txt`, `mutants_vis1b.txt`). The vis1 runs were on the clone of `0b3a6d0` (after the pattern fix of `table_wrong_direction`); the vis2 run on `72adb92`, whose `src` is the same.

| # | Mutant | File | Failing tests | Node |
|---:|---|---|---|---|
| 1 | softmax_over_channels | JiT_T2I.py | T7, T11 | vis2 |
| 2 | c1_c2_swapped | gmflow.py | T1, T3, T4, T5, T10, T12, T13, T14, T29 | vis2 |
| 3 | c3_dropped | gmflow.py | T3, T5, T13, T14 | vis2 |
| 4 | s_x_equals_s_u | gmflow.py | T1, T3, T4, T6, T12, T13, T14, T21, T23, T25, T29 | vis2 |
| 5 | no_bridge_noise | sampling_gmflow.py | T5, T13, T14 | vis2 |
| 6 | spread_input_not_detached | JiT_T2I.py | T11 | vis2 |
| 7 | spread_varies_over_channels | JiT_T2I.py | T11, T14, T26 | vis2 |
| 8 | avg_var_wrong_axis | gmflow.py | T14 | vis2 |
| 9 | k_without_half | gmflow.py | T14 | vis1 |
| 10 | no_first_order_fallback | sampling_gmflow.py | T14 | vis1 |
| 11 | table_wrong_direction | gm_spread.py | T21, T24, T28 | vis1 |
| 12 | s_x_squared_twice_kernel | gm_ode_triton.py | T26 | vis1 |
| 13 | s_x_squared_twice_spread | gm_spread.py | T21, T23 | vis1 |
| 14 | kernel_substep_time_off_by_one | gm_ode_triton.py | T26 | vis1 |
| 15 | bound_guard_off (new: `s_x > 2 * bound`) | gm_spread.py | T28 | vis1 |
| 16 | table_ckpt_unchecked (new: any ms-step accepted) | gm_spread.py | T28 | vis1 |
| 17 | fused_grid_off_by_one (new: weights (j + 1)/(n + 1)) | sampling_gmflow.py | T25, T26 | vis1 |

The first vis1 attempt stopped at mutant 11 with "pattern not found once": my fp64 refactor of `SpreadTable` had moved the interpolation line out of its parentheses. `0b3a6d0` updates the pattern; mutants 11–17 then ran. `mutate.py` now matches every pattern exactly once (checked for all 17).

## What I could not test

1. Nothing ran on a GPU. Compile, DDP, the per-pixel resume, the eval path with the hashed name, the Triton kernel compiled on H100 and the new fp32 benchmark checks wait for the smoke, H1 and N.
2. The 1e-4 tolerance of the fp32 end-to-end check is set from CPU numbers (3.8e-6 at 8 × 16 on the small model). On H100 with the d640 net over 8 calls it may be larger. If the benchmark fails only that check, read the JSON before calling it a kernel bug.
3. The real calibration table is not known yet. If its s_x exceeds (1 − t)/t at t = 0.875 (the last call at 8 calls), H2 at 8 calls is refused by design. The CPU smoke table passes there with margin (0.06 vs 0.143).
4. The second-order code is unchanged and untested on GPU; no job uses it.
5. T14 on vis2: the cdist fix removes the AMD mm-path artifact; the suite ran on vis1 only. Mutants 1–8 ran on vis2 (AMD) and killed as on vis1 before.

## Jobs to submit

`W=/scratch/gpfs/GROUP/USER/tools/wq/wq`, `J=$WC/project/gpic-wt/gmflow_run3/jobs`, after `git worktree add gpic-wt/gmflow_run3 <final commit>`. All jobs are 8-GPU and fetchable. Times are about 1.25 × the expected time. Expected times come from w1173 (training 36 min for 62336 steps at C0/8 on 8 H100s) and its evaluations (3.5–6 min each: predict 150–270 s plus FD 75 s).

Priority: the owner decides. By the standing rule, the smoke is a test (`--prio 10`); the rest is prio 0 unless named main line.

1. **Smoke** (28 min worst case):
   `$W submit --prio 10 --time 00:50:00 $J/smoke_gmflow2.sh`
2. **D2, per-pixel K = 8, d640, C0/8, 62208 steps**, after the smoke (`--after ok:<smoke>`). Training about 40 min + 10 evals × 5 min ≈ 1.5 h:
   `$W submit --partial --time 02:00:00 --after ok:<smoke> $J/train_gmflow.sh 8 d640 640 9 2 62208 c08px w1147 --wt gmflow_run3 --spread pixel 31104:gm_ode:50 62208:gm_ode:50 31104:gm_ode:8 62208:gm_ode:8 31104:gm_ode:4 62208:gm_ode:4 31104:euler_mean:50 62208:euler_mean:50 31104:euler_mean:8 62208:euler_mean:8`
3. **D3, per-pixel K = 1, d640, C0/8, 66432 steps**, same 10 tasks at 33216 / 66432:
   `$W submit --partial --time 02:00:00 --after ok:<smoke> $J/train_gmflow.sh 1 d640 640 9 2 66432 c08px w1147 --wt gmflow_run3 --spread pixel 33216:gm_ode:50 66432:gm_ode:50 33216:gm_ode:8 66432:gm_ode:8 33216:gm_ode:4 66432:gm_ode:4 33216:euler_mean:50 66432:euler_mean:50 33216:euler_mean:8 66432:euler_mean:8`
   The 10 tasks are the K = 16 run's list (w1185, group E: GM-ODE 50, 8, 4 and Euler on the mean 50 and 8, at C0/16 and C0/8), without gm_ode2 / gm_sde2. They pair with w1173 / w1174's GM-ODE and Euler-on-the-mean numbers.
4. **H1, calibration** (about 10 min, limit 45 min):
   `$W submit --time 00:45:00 $J/calibrate_spread.sh`
5. **H2, gm_ode_cal with the raw table at 8 and 4 calls**, after H1, one job per checkpoint (each table belongs to its step; the eval checks it). 2 evals × 5 min each:
   `$W submit --time 00:30:00 --after ok:<H1> $J/evals_gmflow.sh 0 d640 640 9 2 $WC/runs/gpic-speed/point_spread_cal --link $WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640 --wt gmflow_run3 --table $WC/runs/gpic-speed/point_spread_table/033408/table.json 33408:gm_ode_cal:8 33408:gm_ode_cal:4`
   `$W submit --time 00:30:00 --after ok:<H1> $J/evals_gmflow.sh 0 d640 640 9 2 $WC/runs/gpic-speed/point_spread_cal --link $WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640 --wt gmflow_run3 --table $WC/runs/gpic-speed/point_spread_table/066816/table.json 66816:gm_ode_cal:8 66816:gm_ode_cal:4`
   Compare with w1147's Euler at 8 calls (w1175, done) and 4 calls (w1186, queued), and with w1174 (K = 1) GM-ODE (8 calls done; 4 calls in w1188, queued).
6. **N, sampling benchmark** (about 20 min plus the fp32 checks, limit 50 min):
   `$W submit --time 00:50:00 $J/bench_sampling.sh`
7. **G2, sub-step sweep of w1173 at C0/8** (7 evals × 4–6 min ≈ 40 min; all calls × n ≤ 512). New root, w1173 linked, so nothing is written in w1173's directory:
   `$W submit --time 01:00:00 $J/evals_gmflow.sh 8 d640 640 9 2 $WC/runs/gpic-speed/gmflow_k8_d640_substeps --link $WC/runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640 --wt gmflow_run3 62336:gm_ode:8:4 62336:gm_ode:8:8 62336:gm_ode:8:32 62336:gm_ode:8:64 62336:gm_ode:4:8 62336:gm_ode:4:16 62336:gm_ode:4:64`
   The defaults (n = 16 at 8 calls, n = 32 at 4 calls) are w1173's own `eval_062336_gm_ode_calls8` and the 4-call result of the queued group-A evaluation w1187.
8. **L1, λ = 1, K = 8, d640, C0/8, 62336 steps** (training 36 min + 8 evals × 5 min ≈ 1.3 h):
   `$W submit --partial --time 01:45:00 $J/train_gmflow.sh 8 d640 640 9 2 62336 c08lam1 w1147 --wt gmflow_run3 --lam 1 31168:gm_ode:8 62336:gm_ode:8 31168:gm_ode:4 62336:gm_ode:4 31168:gm_ode:50 62336:gm_ode:50 31168:euler_mean:50 62336:euler_mean:50`
9. **L2, λ = 1, K = 1, d640, C0/8, 66560 steps**:
   `$W submit --partial --time 01:45:00 $J/train_gmflow.sh 1 d640 640 9 2 66560 c08lam1 w1147 --wt gmflow_run3 --lam 1 33280:gm_ode:8 66560:gm_ode:8 33280:gm_ode:4 66560:gm_ode:4 33280:gm_ode:50 66560:gm_ode:50 33280:euler_mean:50 66560:euler_mean:50`

Notes on the list:
- `--partial` on the training jobs: they are under 3 h, so the rule does not require it, but `train_gmflow.sh` is resumable (fetchable-partial) and skips finished work, so it costs nothing. The evaluation-only jobs (H1, H2, N, G2, smoke) are under 1 h and not partial.
- No D4 (no second-order evaluations). D2 and D3 run on `gmflow_run3`, whose `spread.txt` / `lam.txt` guards are new; their roots `gmflow_k{8,1}_d640_c08px` and `..._c08lam1` do not exist yet.
- The two λ = 1 runs pair with w1173 / w1174 (λ = 0.5, same steps and data). The judgment (section 11 of `docs/gmflow.md`): λ = 0.5 stays only if it is 5% lower than λ = 1 at both C0/16 and C0/8 for GM-ODE, each number of calls apart; else the method uses λ = 1. D2 / D3 against w1173 / w1174: the per-pixel spread enters the method only if it is 5% lower at both points.
