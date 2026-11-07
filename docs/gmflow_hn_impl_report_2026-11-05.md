# GMFlow-HN implementation report (2026-11-05)

Branch `gmflow-hn`, worktree `gpic-wt/gmflow_hn`. Base: c16d71d. Commits b6f5634..8b6db5c (6 code commits, plus the docs commit). Nothing was pushed, submitted or cancelled. No other worktree was touched.

**Result.** All three additions (H, N, G2) are built and tested on CPU (vis1). The suite has 17 tests: T1–T10 and the new T21–T27. All 17 pass. All 9 mutants are killed, 4 of them new. The two GPU job scripts are written and not submitted, so there are no FD or throughput numbers yet.

## 1. Files changed

| File | Status | What |
|---|---|---|
| `baselines/PixelGen/src/diffusion/flow_matching/gm_spread.py` | new, 40 lines | `SpreadTable` (:12, linear in t, clamped at the ends) and `point_mixture` (:32, x_hat → K = 1 mixture with a given s_x) |
| `baselines/PixelGen/src/diffusion/flow_matching/gm_ode_triton.py` | new, 128 lines | `gm_ode_substeps_ref` (:32, PyTorch reference), `_gm_ode_kernel` (:57–:111, Triton), `gm_ode_substeps_fused` (:113, launcher) |
| `baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py` | +25 −3 | `GMODESampler.__init__` gets `spread_table=None` and `fused=False` (:76–:82). Loop: an assert and `_point_gm` (:91–:92), the fused branch (:97–:99). New methods `_point_gm` (:109) and `_fused_substeps` (:116) |
| `tools/gmflow/calibrate_spread.py` | new, 166 lines | H1: `load_val` (:30), `noise_for` (:55), `error_sums` (:62), `build_net` (:81), `cmd_run` (:93), `merge` (:126), `cmd_merge` (:143) |
| `tools/gmflow/bench_sampling.py` | new, 171 lines | N2: throughput, the per-call check and the end-to-end check |
| `jobs/eval_gmflow.sh` | +16 −1 | header :18–:19; `EVAL_SUBSTEPS` :32–:37; sampler `gm_ode_cal` :43–:48; output name `${NSUF}` :56; table recorded :75 |
| `jobs/evals_gmflow.sh` | +8 −3 | usage :14–:15; `--table` :23; `EVALS_WT` :25 (removed at the merge into gmflow, 2026-11-06: use `--wt <worktree>`); 4th field :44–:47 |
| `jobs/calibrate_spread.sh` | new | 8 GPUs, fetchable, 45 min: w1147's 033408 and 066816 → `runs/gpic-speed/point_spread_table/<step>/table.json` |
| `jobs/bench_sampling.sh` | new | 8 GPUs, fetchable, 50 min: runs `bench_sampling.py` |
| `tools/gmflow/test_gmflow.py` | +285 | T21 (:447), T22 (:470), T23 (:504), T24 (:525), T25 (:588), T26 (:606), T27 (:641) |
| `tools/gmflow/mutate.py` | +5 | 4 new mutants |
| `docs/gmflow_hn.md`, this report | new | |

## 2. Each item of the brief: what was done and why

### H1. `tools/gmflow/calibrate_spread.py`

- **Held-out data.** I used the GPIC validation split. Training reads only `data/gpic/train/gpic_train_*.tar` (`lite_ladder.yaml` and `train_ladder.sh`). The val tars are `data/gpic/val/gpic_val_{00000..00031}.tar`. The script takes the first 4096 captioned images of `gpic_val_00000.tar`, in tar order. That tar holds 6101 pairs, so one tar is enough. A clean held-out split was available, so the training-slice fallback was not needed.
- **Preprocessing.** Images use the val transform: the ADM centre crop to 256, scaled to [−1, 1]. This is what `GPICWebDataset(is_val=True)` uses. Training uses a random crop (see risks).
- **Model.** The EMA weights (`ema_denoiser.*`) go into `JiT_T2I` with the eval's arguments (d640: 10 groups, 9 + 2 blocks, txt_max_length 320, bottleneck 256). The load is strict except for the fixed `pos_embed` buffer. The script asserts a point-model head (768 outputs). Parameters stay fp32 and the forward runs under bf16 autocast, as Lightning's `bf16-mixed` predict does after `on_predict_start` casts the EMA net to fp32. No CFG.
- **Captions.** `Qwen3TextEncoder` in masked mode with pad multiple 8 and SDPA, as in `lite_ladder_sample.yaml`. The encoder runs under bf16 autocast. Each image batch is encoded once and reused at all 64 t.
- **Noise.** A fresh draw per (image, t). Each image has its own generator, seeded `seed·1000003 + global index`, so the result does not depend on how images are split across GPUs (T24 checks this).
- **t grid.** 64 uniform points in [0.01, 0.99].
- **Output.** Each process writes `shard<i>.json` with per-t sums of the per-image MSE and of its square. `merge` checks that the shards come from one run and that none is missing or repeated. It then writes `{"t", "s_x", "mse_x", "se_mse_x", "s_x_trivial", "n_images", "ckpt", "data", "seed"}`. `se_mse_x` is the standard error over images. `s_x_trivial = (1 − t)/t` is the error of the estimator x_t / t. The extra fields are diagnostics; the sampler reads only `t` and `s_x`.
- **8 GPUs.** One plain process per GPU, as `jobs/posterior_mc.sh` does. No DDP. Image i goes to shard i mod 8.
- **CPU.** `--device cpu` works. Two small CPU-only changes were needed: the encoder's `_to_cuda` is bypassed, and the caption embedding is cast to fp32 because CPU runs without autocast. The GPU path does not use either.

### H2. `spread_table` in `GMODESampler`

- At each call, `_point_gm` turns the point model's x_hat into a K = 1 mixture. The mean is x_hat. log s_u = log s_x(t) − log σ, so that `logstd_x` returns s_x. The existing sub-step loop then runs unchanged, so the math is literally the K = 1 code.
- Interpolation is linear in **t**, as the brief asks, and clamps at the ends like `numpy.interp`.
- With s_x = 0, `point_mixture` gives log s_u = −inf. That is exactly the dict `as_mixture` builds, so the sampler is **bitwise** the no-table sampler. It matches the exact step (`ImplicitSampler`) to fp32 rounding (T22: max |diff| 2.4e−7 to 4.2e−7). This holds because `gm_convert` with s_x = 0 returns the mean unchanged, and the Euler sub-steps compose to DDIM.
- A table given with a mixture model raises an assert, because the table is meant for point models only.
- The `gm_spread` import is lazy, so the module imports exactly as before when no table is given.

### H3. `gm_ode_cal` in `eval_gmflow.sh`

- The sampler is accepted only for K = 0 and only with `EVAL_SPREAD_TABLE` pointing to an existing file. The script also checks that the worktree has `gm_spread.py`.
- It uses `gmflow_ode_sample.yaml`, the same sub-step count rule as `gm_ode`, and `--model.diffusion_sampler.init_args.spread_table <json>`.
- Output: `eval_<step>_gm_ode_cal_calls<calls>`. The `_cal` comes from the sampler name. The table path goes into `ckpt.txt` and the table is copied into the output directory, so a result always carries its table.
- The existing guard rejects `gm_ode` for K = 0. I left it unchanged: `gm_ode` on a point model would be the exact step, a duplicate of `euler` at 16 calls or fewer.

### G2. `EVAL_SUBSTEPS`, the 4-field tasks and `--table`

- `EVAL_SUBSTEPS` must be a positive integer and is accepted only for `gm_ode` and `gm_ode_cal`. When it is set, the output directory gets `_n<substeps>`. When it is unset, every name and argument is as before (T27).
- `evals_gmflow.sh` accepts `step:sampler:calls:substeps`. The done-check and the log name include `_n<substeps>`. Three-field tasks are parsed as before. `--table <json>` comes after the optional `--link` and before the tasks.
- Added `EVALS_WT` (default `gmflow_run2`, the old constant; removed at the merge into gmflow on 2026-11-06, where the argument `--wt <worktree>` replaces it). Reason: `gm_ode_cal` needs code that `gmflow_run2` does not have. Note risk 6: a fetchable job cannot take environment variables from the submitting shell.

### N1. `gm_ode_triton.py`

- **One launch per network call** for all n sub-steps. Grid (ceil(HW / 256), B). Each program holds 256 pixels of one image with all K components in registers (K padded to a power of 2; padded components get log-weight −inf). Means and log weights are read once per call; x is read once and written once. All math is fp32.
- **Algorithm.** The same as `GMODESampler`'s loop with `gm_convert` written out per pixel. Sub-step 0 uses the mixture mean. Sub-step j > 0 re-weights the mixture to the current x (Eq. 10), using the same clamps as `gm_convert` (1e−6 on denom and norm). Each sub-step is an Euler step on the uniform σ grid. The sampler builds the σ grid exactly as its loop does.
- **s_x shapes.** (B, 1, 1, 1, 1) and (B, 1, 1, H, W), selected by strides. The per-pixel shape is for the other worker's per-pixel spread.
- **Sub-step count.** A `tl.constexpr`, so there is one compile per count (16 at 8 calls, 32 at 4 calls). I first passed it as a runtime argument, and the Triton 3.3.1 interpreter could not loop over a runtime integer.
- **Integration.** `GMODESampler(fused=True)` calls `_fused_substeps`, and the PyTorch loop runs 0 times. The default `fused=False` leaves the loop unchanged. The trajectory output (`v_trajs`) is computed the same way in both paths.
- **Interpreter.** Triton has a CPU interpreter (`TRITON_INTERPRET=1`). T26 re-runs itself in a subprocess with that variable set and runs the real kernel on CPU.

### N2. `tools/gmflow/bench_sampling.py`, `jobs/bench_sampling.sh`

- torchrun with 8 processes. Each rank takes its own 1024 prompts from `gpic_test_50k.jsonl` with fixed noise, batch 32. Captions are encoded once before any timing; the text encoder is the same for every sampler.
- Nets: EMA fp32, `net.compile()` (as `configure_model`), bf16 autocast around the sampler. This is the eval path without Lightning.
- Settings: point model (w1147 066816) with the official Euler at 50, 16, 8 and 4 calls; K = 8 (w1173 062336) with GM-ODE at 8 and 4 calls, unfused and fused.
- Each setting: one warm-up pass, which compiles every caption length. Then timed passes until the slowest rank has run at least 60 s. images/s = passes × 1024 × 8 / the slowest rank's time.
- **Checks.** (1) Per call: fused against the reference on a real K = 8 output, fp32; it must agree to 1e−4. The script also times the sub-steps of one call alone, in ms. (2) End to end: fused against unfused on the same prompts and noise; the mean absolute difference must be under 5e−3 and fewer than 2% of uint8 pixels may differ by more than 1. The job exits 1 if a check fails.
- Output: `runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640/bench_sampling_<mmdd_HHMM>.{json,log,commit.txt}`.
- **Time budget.** 8 settings × (60 s or more of timing + warm-up + compile), plus model loads: I expect about 20 min. The limit is 50 min, and a `timeout` of 45 min sits inside it.

### Job script `jobs/calibrate_spread.sh`

- 8 GPUs, `-c 96`, 45 min, `--comment=fetchable`, `#FETCHABLE`.
- It sources `env.sh`, which sets TMPDIR and the proxy, and uses the constant `WT=$WC/project/gpic-wt/gmflow_hn_run`. It checks that the worktree holds the new code and has no local changes, and records the commit.
- Each checkpoint (033408, 066816) runs as 8 processes on 8 GPUs, then merges. A finished table is skipped, so a rerun does only what is missing.
- Expected run time is about 10 min. 4096 images × 64 t is 262k forwards of d640, about 32k per GPU.

## 3. Test output (vis1, CPU, `tools/gmflow/run_vis.sh tools/gmflow/test_gmflow.py`, 31 s)

```
T1 PASS  (ratio 6.122448 vs 6.122449)
T2 PASS
T3 PASS
T4 PASS
T5 PASS
T6 PASS
T7 PASS  (gm_k = 0 bit-identical to 9711ff9)
T8 PASS
T9 PASS  (K=8 r 0.07010, K=1 r 0.00245)
T10 PASS
T21  table: linear between points (= numpy.interp), clamped beyond the ends; point_mixture returns s_x; s_x = 0 is log s_u = -inf      PASS
T22  calls 8, n 16: = no table bitwise; max |diff| to the exact step 2.98e-07
     calls 4, n 32: = no table bitwise; max |diff| to the exact step 4.17e-07
     calls 5, n 3:  = no table bitwise; max |diff| to the exact step 2.38e-07                                                      PASS
T23  calls 8, n 16: rms error to the exact ODE: exact step 0.08551, calibrated 0.00578, table s_x^2 0.07096, s_x = 0 0.08551
     calls 4, n 32: rms error to the exact ODE: exact step 0.16001, calibrated 0.00578, table s_x^2 0.11277, s_x = 0 0.16001
     calls 2, n 64: rms error to the exact ODE: exact step 0.30104, calibrated 0.00577, table s_x^2 0.16041, s_x = 0 0.30104     PASS
T24  s_x got  0.5014 0.4974 0.4483 0.2237 0.0101
     s_x true 0.5000 0.4961 0.4472 0.2236 0.0101
     load_val: 3 images from gpic_val_00000.tar
     merge over 3 shards = one pass (rel. diff <= 1e-12)                                                                        PASS
T25  reference = the sampler's loop (K 1, 3, 8; s_x per image and per pixel; 3 calls x 7 sub-steps)                             PASS
T26  kernel = reference (K 1, 3, 8; s_x per image and per pixel; pixels not a multiple of the block; sigma from 1 to 0)
     sampler fused = not fused (K = 8 mixture; point model with a table)                                                        PASS
T27  eval_gmflow.sh: default names unchanged; gm_ode_cal needs the table and K = 0; EVAL_SUBSTEPS adds _n<n> and only for GM-ODE
     evals_gmflow.sh: 3-field tasks unchanged; a 4th field sets EVAL_SUBSTEPS and the _n<n> name; --table sets EVAL_SPREAD_TABLE   PASS
ALL PASS
```

What the new tests check:
- **T23, the toy.** x0 ~ N(m, c²I) with c = 0.5. The network returns the exact posterior mean, and the posterior std is known in closed form. The exact probability-flow ODE maps eps to m + c·eps, because the flow is affine. With the true std as its table, the calibrated GM-ODE has error 0.0058 at 2, 4 and 8 calls. That error is only the sub-step discretization. The exact step's error is 0.086 to 0.30, so the calibrated GM-ODE is 15 to 52 times closer. A wrong table (s_x² in place of s_x) loses most of the gain, so the test needs the correct spread, not just any spread. The test asserts calibrated < 0.1 × exact step and calibrated < 0.5 × the s_x² table, and that s_x = 0 equals the exact step.
- **T24.** `error_sums` with the exact posterior mean on 1500 toy images recovers the true posterior std within 2% at 5 values of t. Three shards merge to the single-pass result within 1e−12 relative. A missing shard makes `merge` raise. `load_val` reads real val images, and the table round-trips through `SpreadTable`.
- **T25 and T26.** The reference equals the sampler's loop (atol 2e−5). The kernel, run in the interpreter, equals the reference (atol 1e−5) for K = 1, 3 and 8, both s_x shapes, HW not a multiple of the block, and σ intervals from 1 down to 0. `fused=True` equals `fused=False` for a K = 8 mixture and for a point model with a table.
- **T27.** Runs stubbed copies of both shell scripts: env.sh is replaced, and a fake worktree and checkpoints are used. It checks output names, sampler arguments, error exits, 3- and 4-field task parsing, `--table`, and skipping of finished `_n` tasks.

**Mutation tests** (`tools/gmflow/run_vis.sh tools/gmflow/mutate.py`, 3 min 51 s):

```
softmax_over_channels           -> killed (its line of failing tests was cut from my captured output; the summary line covers it; docs/gmflow.md 4.3 lists T7)
c1_c2_swapped                   -> ['T1', 'T3', 'T4', 'T5', 'T10']
c3_dropped                      -> ['T3', 'T5']
s_x_equals_s_u                  -> ['T1', 'T3', 'T4', 'T6', 'T21', 'T23', 'T25']
no_bridge_noise                 -> ['T5']
table_wrong_direction           -> ['T21', 'T24']        (interpolation weight (t1 - q) in place of (q - t0))
s_x_squared_twice_kernel        -> ['T26']               (v = s_x^4 in the kernel)
s_x_squared_twice_spread        -> ['T21', 'T23']        (log s_x^2 in point_mixture)
kernel_substep_time_off_by_one  -> ['T26']               (s = sigma[j - 1] in the kernel's sub-step j)
every mutant killed
```

The mutants are applied to a temporary copy of `src`, so the worktree is never edited, and nothing needs to be reverted.

**Real-checkpoint smoke (CPU).** `calibrate_spread.py run` on w1147's `ms-step=066816.ckpt`, 8 val images in 2 shards, then `merge`:

```
shard 0/2: 4 images; s_x at t = 0.01, 0.5, 0.99: 0.5451 0.1562 0.0629
shard 1/2: 4 images; s_x at t = 0.01, 0.5, 0.99: 0.4842 0.1305 0.0444
table: 8 images; s_x(0.01) 0.5156, s_x(0.99) 0.0545
t   [0.01, 0.15, 0.29, 0.43, 0.57, 0.71, 0.85, 0.99]
s_x [0.5156, 0.2646, 0.2012, 0.158, 0.1264, 0.0977, 0.0715, 0.0545]
```

This shows that the checkpoint loads, the text path runs and the table is well formed. The numbers come from 8 images and are not a result.

## 4. What I could not test

1. **The compiled Triton kernel on GPU.** CPU has only the interpreter, which tests the math, not compiled numerics, speed or register pressure. With K = 8 and a block of 256 the kernel holds about 4 × 8 × 256 fp32 values in registers per program. If the job shows spills, use `block=128`. `bench_sampling.sh` measures speed and checks the result.
2. **Throughput numbers.** They need `bench_sampling.sh` on 8 H100s.
3. **The 8-GPU calibration run** and the real tables. They need `calibrate_spread.sh`.
4. **`gm_ode_cal` through Lightning `main.py predict`.** The jsonargparse path of the new `spread_table` argument (a `str` with default `None`) and `fused` (a `bool`) was not run, because predict needs a GPU. The sampler class itself is tested. The CLI flag name follows the existing `substeps` pattern.
5. **The end-to-end bf16 tolerance** of the fused check. It is a guess: mean diff under 5e−3 and under 2% of uint8 pixels off by more than 1. Fused and unfused differ only by fp32 rounding in the sub-steps, but the bf16 network can amplify that along 4 or 8 calls. The per-call fp32 check (1e−4) is the strict one.

## 5. Open risks

1. **s_x is the model's total error, not the posterior std.** An imperfect model's error exceeds the true posterior spread, so the table may overstate it. T23 shows that a wrong spread loses most of the gain. A negative FD result for `gm_ode_cal` would therefore not prove that a learned distribution is needed. ~~Scaling the table (e.g. 0.5×, 0.7×) would be the natural follow-up.~~ Withdrawn (review F3; owner 2026-11-06, 「方法要干净整洁」): a scale picked on test FD is tuned on the test set. There are no scaled or clipped tables. gm_ode_cal runs only with the raw table at 8 and 4 calls, and the sampler refuses a table with s_x(t) > (1 − t)/t at any call (docs/gmflow.md section 10).
2. **One s_x per t for all images and pixels.** The true posterior spread varies a lot across pixels and images, while GMFlow's K = 1 learns s_u per sample. The kernel already supports per-pixel s_x.
3. **Crop mismatch.** The val centre crop differs from the training random crop. The effect on MSE_x is likely small, but it is not measured.
4. **The table is interpolated in t, not σ or log-SNR.** Near t = 0.99 the table changes fast (s_x 0.07 → 0.05 over the last 0.14 of t). Linear interpolation between 64 points spaced 0.0156 apart is fine there. Calls at t < 0.01 (t = 0 at the first call) take the end value s_x(0.01).
5. **Merge conflicts with branch `gmflow`.** `sampling_gmflow.py` has `GMODESampler.__init__` (:76–:82) and the start of its loop (:91–:99) changed. Everything else is in new files or in new, local lines of the two job scripts.
6. **(Resolved at the merge, 2026-11-06: `--wt <worktree>` replaces `EVALS_WT`; the rest of this item is history.)** **`EVALS_WT` is an environment variable,** and a fetchable job cannot take one from the submitting shell. A `gm_ode_cal` evaluation job must set it inside its own script, e.g. a small wrapper that exports `EVALS_WT=$WC/project/gpic-wt/gmflow_hn_run` and calls `evals_gmflow.sh ... --table <json> ...`. I did not write that wrapper, because which evaluations to queue is the owner's call.
7. **The frozen worktree `gpic-wt/gmflow_hn_run` does not exist yet.** Both job scripts exit 2 with "holds old code" until it is created at a commit of this branch.
