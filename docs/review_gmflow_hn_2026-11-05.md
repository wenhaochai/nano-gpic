# Adversarial review of branch gmflow-hn (c16d71d..ffa4bed), 2026-11-05

Reviewer: an independent session. I read every changed file, ran the suite and the mutants on vis1, and ran my own adversarial checks. The check scripts are outside the branch: `/scratch/gpfs/GROUP/USER/tmp/review_hn/adv.py` and `substeps.py`. I simulated the merge with branch gmflow (251b3fa) in a throwaway clone under the same directory. I changed no code on the branch.

**Summary.** The math of H and N is correct. The default paths are bitwise unchanged. The kernel equals the reference on 432 adversarial cases. `calibrate_spread.sh` and `bench_sampling.sh` can run once the frozen worktree exists. The `gm_ode_cal` evaluations have three problems to fix first. (1) The output directory does not name the table, and nothing checks that the table belongs to the evaluated checkpoint. A second table therefore collides silently with the first. (2) The table exceeds a provable upper bound on the posterior std for t > about 0.94. The cause is the model's error floor. This matters at 50 calls. (3) Table scaling is a tuning knob. The arms must be fixed before the run, or chosen on the val split. The merge with gmflow conflicts in 5 files. The resolutions are in section 5.

## 0. What I ran

| Check | Where | Result |
|---|---|---|
| `run_vis.sh tools/gmflow/test_gmflow.py` | vis1 | T1–T10, T21–T27: ALL PASS |
| `run_vis.sh tools/gmflow/mutate.py` | vis1 | 9 of 9 mutants killed (same lists as the report) |
| `adv.py` §1: new `GMODESampler` against the c16d71d `GMODESampler` (loaded from `git show c16d71d:...`), K = 8, K = 1 and point, calls 50/8/4/2 with n = 3/16/32/64, x-dependent nets | vis1 | `x_trajs` and `v_trajs` bitwise equal in all 12 cases |
| `adv.py` §2: kernel (TRITON_INTERPRET=1) vs `gm_ode_substeps_ref` vs a loop built from `gmflow.gm_convert_mean`; K ∈ {1, 8, 16}; logit scale ∈ {0, 1, 1e2, 1e4}; s_x ∈ {1e-6, 1e-3, 0.05, 1, 1e3, per-pixel log-uniform 1e-6..1e3}; σ intervals 1→0.999 (n 4), 1→0.875 (16), 0.5→0.25 (32), 0.02→0 (3), 0.001→0 (8), 1→0 (128); block 16 on 5×7 pixels | vis1 | 432 cases. All are finite. Worst relative difference 4.0e-6 |
| `adv.py` §3: Gaussian toy, table = k × the true posterior std, and a variance floor | vis1 | Table in section 7 |
| `adv.py` §4, `substeps.py`: s_x = 0 against the exact step at large sub-step counts | vis1 | Breaks when calls × n ≥ about 1000 (finding F5) |
| `adv.py` §5: jsonargparse `add_subclass_arguments` with `--s.init_args.spread_table <path> --s.init_args.fused true` | vis1 | Instantiates. Defaults give `spread None, fused False`. This closes report §4 item 4 for the CLI layer |
| Merge simulation: `git merge --no-commit gmflow-hn` into gmflow 251b3fa | throwaway clone | Conflicts in 5 files (section 5) |

## 1. H math

1. **K = 1 at each call: yes.** `_point_gm` (sampling_gmflow.py:109-114) calls `point_mixture(x_hat, spread(t_cur), 1 - t_cur)`. In gm_spread.py:37-40 the means are `x_hat[:, None]` (K = 1), the log weights are 0 and `logstd = log s_x − log max(σ, EPS)`. The sampler reads it back through `logstd_x` (gmflow.py:31-33), which adds `log max(σ, EPS)` at the same σ. So s_x comes back exactly (T21 checks this). The existing sub-step loop then runs unchanged. The code path is the K = 1 code path.
2. **Units: correct.** s_x is a per-dimension std in the [−1, 1] pixel range. `PixelAE` has scale 1 and shift 0, so the sampler's x lives in the same range as the calibration targets (calibrate_spread.py:48, `/127.5 − 1`, equal to the val transform's `ToTensor` + `Normalize(0.5, 0.5)`, gpic_dataset.py:26-30).
3. **The t of the table: the call's t, not the sub-step's.** `self.spread(t_cur...)` is evaluated once per call (sampling_gmflow.py:92). `gm_convert` then carries the variance analytically through the sub-steps (gmflow.py:79). This matches K = 1 GMFlow, where s_x is the posterior std given x at the call.
4. **s_x = 0 gives the exact step.** `log(0) = −inf` gives the same dict as `as_mixture`, so the result is bitwise the no-table path (T22). That path matches `ImplicitSampler` to within 4.2e-7 (T22). It holds only for calls × n below about 1000 (F5). The default counts (≤ 128 total) are inside that range.
5. **Interpolation: correct at both ends.** gm_spread.py:26-29 clamps q to [t_0, t_last] and uses `searchsorted(right=True).clamp(1, len − 1)`. At q = t_last it returns s_last. At q = t_0 it returns s_0. Outside the grid it returns the end values (T21 checks against `numpy.interp` on 1000 points). The first call (t = 0) reads s_x(0.01). That is the right limit: at t ≤ 0.01 the posterior is the data marginal.

## 2. Calibration

1. **Per-dimension MSE in [−1, 1]: yes** (calibrate_spread.py:75, `.flatten(1).mean(1)` over 3·256·256).
2. **EMA weights: yes** (:84, `ema_denoiser.` prefix). The load is strict except `pos_embed` (:88-89). The point-head check is at :85.
3. **Model construction: matches w1147.** The training config `runs/gpic-speed/point_nodrop_d640/exp_ladder_d640/config-TrainerFn.FITTING-2611050106.yaml` has num_groups 10, hidden 640, blocks 9 + 2, txt_max_length 320, bottleneck 256 and gm_k 0. calibrate_spread.py:86-87 builds exactly that.
4. **Text handling: matches the eval, not the training.** Training read cached Qwen3 outputs (`CachedTextConditioner`). The eval and calibrate_spread.py:100 use the online `Qwen3TextEncoder`: masked mode, pad multiple 8, SDPA, bf16 weights, and `BaseConditioner.__call__` casts to bf16. This is correct: the sampler sees eval-path embeddings.
5. **No CFG, conditional branch: yes.** Only `y` and `y_mask` are passed (:117), never the unconditional embedding.
6. **bf16: yes.** fp32 parameters under bf16 autocast (:116), as Lightning `bf16-mixed` after `on_predict_start` casts the EMA net to fp32 (lightning_model_gpic.py:178-179). The eval also compiles the net and the calibration does not. The numerical difference is negligible.
7. **Fresh noise per image and per t: yes** (:55-58, :71). Each image has its own generator, so the result does not depend on the sharding (T24).
8. **t convention: yes.** `x_t = t x + (1 − t) eps` (:73), the same as `JiTTrainer` (training_JiT.py:47) and the samplers.
9. **Held out: yes.** w1147 trained on `data/gpic/train/gpic_train_{00000..00800}.tar` (its config, line 247-248). The calibration reads `data/gpic/val/gpic_val_00000.tar`. The val loop of training computes losses only. Two small mismatches remain. (a) The training crop is random (0.8–1.0 of the short side, augmentation.py:57) and the calibration crop is the centre crop. The calibration therefore sees slightly larger-scale content. (b) The caption-type mix of the first 4096 val images is 43% short, 43% medium, 13% long, 1.4% tag. The 50K test prompts are 45% / 45% / 9% / 1%. Both effects are small. Neither is measured.
10. **Reduction over 8 processes: correct.** Image i goes to shard i mod 8 (:97). Each shard sums per-image MSE over all its batches, the partial last batch included (:68). `merge` divides the summed sums by the summed counts (:132-135). It asserts that the shard ids are complete and not repeated (:131). There is no double counting and no dropped-batch bias. 4096 / 8 / 32 gives exactly 16 batches per shard. One gap: `merge` does not compare the `images` field across shards (:130). The job removes old shards first (calibrate_spread.sh:200), so this gap cannot bite there (LOW).
11. **Precision of the table.** From the code's own smoke numbers (two shards of 4 images, report §3), the per-image MSE has a coefficient of variation of about 0.33 at t = 0.01, 0.5 at t = 0.5 and 0.95 at t = 0.99. These are rough single-pair estimates. With N = 4096 the standard error of the mean MSE is CV/64: 0.5%, 0.8% and 1.5%. On s_x it is half of that: 0.3%, 0.4% and 0.7%. The same images serve every t, so the error is mostly a smooth common factor, not jitter from t to t. The toy (section 7) needs a 1.5× error in s_x before the result changes much. So 4096 images and 64 t values are enough. `se_mse_x` in the table measures this directly. Check it after the run.
12. **Coupling (LOW).** `error_sums` takes `ts`, but `noise_for` draws `len(T_GRID)` noises (:58, :71). A call with another grid breaks. T24 patches the global to work around this. Production uses `T_GRID`, so the job is not affected.
13. **Job (calibrate_spread.sh).** It follows the guards: env.sh, `#FETCHABLE`, `--comment=fetchable`, `-c 96`, the frozen `WT`, the clean-tree check, 8 GPUs, a per-process `timeout 1200`. Its worst case of 2 × 20 min fits the 45 min limit. Expected work: 262k d640 forwards per checkpoint, against 400k forwards in 193 s for an 8-call eval (w1173 `phase_times.json`). The 10-min estimate is plausible. Two defects. (a) `commit.txt` sits at the root and is overwritten on every run (:188). If the two tables come from different starts, the file names only the last commit (LOW). (b) The frozen worktree `gpic-wt/gmflow_hn_run` does not exist. Both jobs exit 2 until it is created (precondition).

## 3. Triton kernel (N1)

Line-by-line comparison of gm_ode_triton.py:98-150 with sampling_gmflow.py:99-103 and gmflow.py:71-88:

| Item | Sampler loop / gm_convert | Kernel | Same? |
|---|---|---|---|
| σ grid | `σ_src + (σ_end − σ_src)·(j/n)` (:100-101) | `_fused_substeps` stacks the same expression (:121) and the kernel loads `sig[j]`, `sig[j+1]` | yes, bitwise the same σ values |
| sub-step 0 | `gm_mean` = Σ exp(lw)·μ | `tl.sum(exp(lw)·μ_c)` (:120-124) | yes |
| a, b | `ats = (1−σ_t)σ_s`, `ast = (1−σ_s)σ_t` | `a = (1−s)s0`, `bb = (1−s0)s` (:128-129) | yes |
| denom | `max(ats² − ast², 1e-6)` | `max(a² − bb², 1e-6)` (:130) | yes |
| g, g_var | (:77-78) | (:131-134), with x_src = the state at the call | yes |
| v | `exp(2 logstd_x(σ_src))` | `sx²`, with sx = `logstd_x(gm, σ_src).exp()` from the launcher (:122) | yes |
| norm | `max(g_var + v, 1e-6)` | (:135) | yes |
| weights | `log_softmax(lw − ‖μ−g‖²/(2 norm))` then exp | max-subtracted softmax (:139-141); padded components get −inf (:111) | yes up to rounding |
| new means | `(g_var μ + v g)/norm` | (:142-144) | yes |
| Euler | `x + (s'−s)(x−x0)/s` | the same order of operations (:145-147) | yes |
| last sub-step (s' = 0) | gives x0 | the same; s is never 0 because only s' reaches 0 | yes |
| precision | fp32, autocast off | all loads `.to(float32)`; the launcher passes fp32 x and s_x | yes |
| s_x shapes | (B,1,1,1,1) or (B,1,1,H,W) | strides (1, 0) or (HW, 1) (:162) | yes; both are tested |

My 432 adversarial cases (section 0) cover extreme logits (scale 1e4, one-hot weights), s_x from 1e-6 to 1e3, t near 0 (σ = 1 → 0.999) and near 1 (σ = 0.001 → 0), and K = 1, 8, 16 with HW not a multiple of the block. All outputs are finite. The worst relative difference among kernel, reference and the `gm_convert_mean` loop is 4.0e-6.

**Compile-time sub-step count.** It is not a problem. The eval's `ceil(128/calls)` takes 5 values (3, 8, 16, 32, 64). Each value compiles once and is cached in `TRITON_CACHE_DIR` (env.sh). The loop over the `constexpr` bound stays an `scf.for`, so a large n does not grow the code. Also, `eval_gmflow.sh` never sets `fused`. Only the benchmark uses the kernel.

**Performance notes (LOW).** (a) `_fused_substeps` builds the σ grid with n + 1 small device ops per call (:121), which is 65 tiny launches at n = 64. This partly cancels the "one launch per call" gain. Build the grid on the host, or with one `torch.linspace`. (b) With K = 8 and BLOCK 256, the 4 × 8 × 256 fp32 register tiles plus temporaries can spill. The report already says this. The benchmark measures it.

## 4. Benchmark (N2)

1. **Times only sampling: yes.** Captions are encoded before any timer (bench_sampling.py:67-76). The timed loop contains only `run_pass` (:103-111).
2. **Warm-up: yes.** One full pass per setting (:98-100) compiles every caption length.
3. **Synchronization: yes.** `cuda.synchronize` + `barrier` before t0 (:98, :100). `synchronize` + `all_reduce(MAX)` of the elapsed time after each pass (:107-109). The throughput is the slowest rank's.
4. **Enough prompts.** 1024 per GPU, at least 60 s per setting. At 50 calls one pass takes about 150 s, so that setting gets 1 timed pass. That is enough. The total, about 20 min, fits the 45-min `timeout`.
5. **Checkpoints: right.** Point `point_nodrop_d640/.../ms-step=066816.ckpt` (C0/8). K = 8 `gmflow_k8_d640/.../ms-step=062336.ckpt` (C0/8). Both exist.
6. **Official Euler: yes.** `EulerSamplerJiT` with the constructor defaults (`ode_step_fn`, `last_step` None, guidance interval [0, 1], timeshift 1, w_scheduler Linear) equals `lite_ladder_sample.yaml`'s sampler (bench_sampling.py:46-48).
7. **The fused-vs-unfused end-to-end check is weak in bf16 (MEDIUM).** The two runs differ only by about 1e-6 fp32 rounding in the sub-steps. The bf16 network can amplify that over 4 or 8 calls to an unknown size. A threshold of mean |diff| < 5e-3 (:158) can fail without a bug, or pass with a small real error (for example a 1e-3 bias on one sub-step). The strict check is the per-call one (:121-131). That check uses only the first call (σ = 1 → 1 − 1/calls). It never exercises the last call (σ → 0), where the clamps act. On a check failure the job still writes the JSON before it exits 1 (:160-167), so no throughput data is lost. Fix: add a per-call check at the last call (t = 1 − 1/calls). Make the end-to-end check fp32 (autocast off) on 32 prompts with a tight tolerance (1e-4). Keep the bf16 numbers as a report only.
8. **Init order (LOW).** `init_process_group("nccl")` runs before `torch.cuda.set_device` (:57-59). The later barriers run after `set_device`, so this works on current PyTorch. Set the device first anyway.

## 5. G2, the job interface, and the merge with gmflow

**G2 (eval_gmflow.sh:31-37, :56).** `EVAL_SUBSTEPS` must be a positive integer. It is accepted only for `gm_ode` and `gm_ode_cal`. It adds `_n<n>`. Unset, it changes nothing: `NSUF=""`, and `SUBSTEPS` uses the old formula. A 4-field task parses correctly (evals_gmflow.sh:43-44). A 3-field task passes `EVAL_SUBSTEPS=` (empty), which `-n` treats as unset (T27). A 5-field task is rejected by the regex.

**Existing calls are unchanged.** `git diff c16d71d..ffa4bed -- jobs/` shows only added lines, one changed `OUT=` line that appends an empty `${NSUF}`, and in evals the `WT=` constant changed to `${EVALS_WT:-<same constant>}`. All existing 3-field calls, samplers and directory names stay the same. T27 checks `euler` and `gm_ode` names and arguments.

**Defects in the job interface:**

- **F1 (HIGH): the gm_ode_cal result is not tied to its table.** The output is `eval_<step>_gm_ode_cal_calls<calls>` (eval_gmflow.sh:56). It does not name the table. A second table for the same step and calls (a scaled table, the bound-clipped table of F2, or the other checkpoint's table) hits `metrics.json exists` (:57). `evals_gmflow.sh:45` reports this as "done before" and exits 0. The old number is then read as the new arm's result. Also, nothing checks that the table belongs to the evaluated checkpoint: `--table .../033408/table.json` with task `66816:gm_ode_cal:8` runs without an error. The table records `ckpt` (calibrate_spread.py:139), but nothing reads it. Fix: in eval_gmflow.sh, require that `basename(table["ckpt"])` contains `ms-step=$STEP`. Put a table tag in the name, for example `_cal-<sha1 of the table, 8 hex>` or a `tag` field written by `merge`. Mirror the same name in the evals done-check.
- **F6 (MEDIUM): `EVALS_WT` is an environment variable** (evals_gmflow.sh:25). A wq job runs in a clean login environment, and nothing comes from the submit shell (tools/wq/README.md rule 3). So a gm_ode_cal list cannot be queued without a wrapper script. Branch gmflow solved the same problem with the argument `--wt <worktree>`. At the merge, drop `EVALS_WT` and use `--wt gmflow_hn_run` (or the merged frozen worktree).
- **F8 (LOW): a relative `--table` path breaks.** The `-f` check runs before `cd $WT/baselines/PixelGen` (eval_gmflow.sh:45 vs :59). `cp` (:75) and `main.py` then resolve the path from the new directory and fail. The failure is loud. Fix: `EVAL_SPREAD_TABLE=$(realpath ...)` in both scripts.

**Merge with gmflow.** I merged gmflow-hn into gmflow at 251b3fa (gmflow's committed work: per-pixel spread, GM-SDE 2 / GM-ODE 2, `--wt`, `--spread pixel`). Git reports conflicts in exactly these hunks:

| File | Hunk | Resolution |
|---|---|---|
| `src/diffusion/flow_matching/sampling_gmflow.py` | GMODESampler docstring end + `def __init__` (gmflow adds the "order 2" sentence; hn adds `spread_table`, `fused`) | Keep gmflow's docstring and hn's signature `(self, *args, substeps=3, spread_table=None, fused=False, **kwargs)`. The loop auto-merges to `_point_gm` → `order 2` → fused/unfused sub-steps, which is the right order. Add `assert not (self.spread is not None and self.order == 2)`. gmflow's docstring says order 2 on a point model changes nothing. That stops being true with a table: the surrogate variance becomes s_x² > 0. Such a sampler is new and untested. |
| `jobs/eval_gmflow.sh` | the `echo ... > ckpt.txt` line (gmflow adds `spread=$SPREAD` and the `SPREAD_ARGS` line; hn adds the `spread_table` line) | Keep gmflow's echo and its `SPREAD_ARGS` line, then hn's `[ $SAMPLER != gm_ode_cal ] || {...}` line. The rest auto-merges: gmflow's `case $K:$SAMPLER` guard replaces the old `if`, and hn's `NSUF` block and `gm_ode_cal` case stay. Two semantic fixes follow. (a) Extend `EVAL_SUBSTEPS` to `gm_ode|gm_ode2|gm_ode_cal`. (b) Add `gm_ode_cal` to gmflow's usage line (line 12). |
| `jobs/evals_gmflow.sh` | header usage + Code comment | One usage line: `[--link <run dir>] [--wt <worktree>] [--spread pixel] [--table <json>] <step:sampler:calls[:substeps]> ...`. Keep gmflow's `--wt` sentence and hn's 4th-field / `--table` sentence. |
| `jobs/evals_gmflow.sh` | option parsing (`--wt`, `--spread` vs `--table`, `EVALS_WT`) | Keep `WTN`, `--wt` and `--spread`. Add the `--table` line after `--spread`. Use `WT=$WC/project/gpic-wt/$WTN` and drop `EVALS_WT`. Also update the `[ $# -ge 7 ]` usage echo (auto-merged to gmflow's version) to name `--table`. |
| `tools/gmflow/mutate.py` | the two new mutant blocks | Keep both blocks. No key collides. |
| `tools/gmflow/test_gmflow.py` | one block after T10 (gmflow T11–T14, hn T21–T27) | Keep both blocks, T11–T14 first. No helper name collides: gmflow adds no top-level `_` helpers. T27 must then drive evals with `--wt <name>` under a stub `$WC/project/gpic-wt/<name>` instead of `EVALS_WT`. Add one test that runs `fused=True` on a per-pixel `logstd` from the merged `JiT_T2I` (gm_spread pixel), and one with `order=2`. |

`train_gmflow.sh` does not conflict: hn did not touch it.

## 6. Defaults

With `spread_table=None` and `fused=False`, the new sampler equals the c16d71d sampler bitwise. This holds for K = 8, K = 1 and the point model at 50/8/4/2 calls, for both trajectories (`adv.py` §1). `GMSDESampler` and `as_mixture` are untouched. T1–T10 pass. T7 still asserts `gm_k = 0` bit-identical to 9711ff9. The new modules are imported lazily, so `sampling_gmflow` imports as before.

**F5 (MEDIUM, pre-existing in gmflow.py:80, copied in the kernel at gm_ode_triton.py:89/:135): the clamp `norm ≥ 1e-6` breaks GM-ODE at many sub-steps with a small s_x.** At the last call, the sub-step before σ = 0 has σ = 1/(calls·n). Then g_var ≈ σ². Once g_var + s_x² < 1e-6, the clamp scales every mean by g_var/1e-6 < 1. Measured with s_x = 0 against the exact step (`substeps.py`). At 8 calls: rms 1.9e-6 at n = 96, 3.5e-2 at n = 128, 0.46 at n = 192. At 4 calls: 1.7e-6 at n = 192, 3.4e-2 at n = 256. At 50 calls, n = 300: max 3.0. The default counts (calls·n ≤ 128) are safe. Calibrated tables (s_x ≥ about 0.05) are safe. An `EVAL_SUBSTEPS` sweep on a model with a small learned s_x at the last call is not safe, and T22 tests only calls·n ≤ 128. Fix: lower the clamp to 1e-20, or return the means unchanged where v = 0. Until then, reject `CALLS × EVAL_SUBSTEPS > 512` in eval_gmflow.sh.

## 7. Interpretation risk

The table holds E‖x − x̂‖²/D = E[Var(x | x_t)] + E‖E[x | x_t] − x̂‖². This is an upper bound on the mean posterior variance. There is a second, model-free upper bound: the error of the estimator x_t / t, s_trivial(t) = (1 − t)/t. The table records it (calibrate_spread.py:139), but nothing uses it.

**F2 (HIGH for 50-call arms, MEDIUM otherwise): for t > about 0.94 the table exceeds the provable bound.** The 8-image smoke gives s_x(0.99) = 0.055. The bound is s_trivial(0.99) = 0.0101, so the table is 5× the largest possible posterior std there. The curves cross where (1 − t)/t ≈ 0.06, near t = 0.94. The excess is the model's error floor, and that floor is bias, not uncertainty. A possible cause is the patch bottleneck (768 → 256 dims, `bottleneck_dim: 256`), which may stop the model from copying x_t at high t. I did not test this cause. The calls at 8, 4 and 2 steps stop at t = 0.875, 0.75 and 0.5, which are below the crossing. The 50-call grid has calls at t = 0.94, 0.96 and 0.98, which are above it. Fix (no free parameter): have the sampler use `min(s_x, s_trivial)`. Write it as a second column in `merge`, or as a second table arm fixed now.

**What the toy says about a wrong spread** (`adv.py` §3: Gaussian x, exact posterior mean as the "model", rms error to the exact ODE solution):

| calls (n) | exact step | 0.5× | 0.7× | 1.0× (true) | 1.5× | 2× | 3× | √(std² + 0.05²) | √(std² + 0.1²) |
|---|---|---|---|---|---|---|---|---|---|
| 50 (3) | 0.0147 | 0.0120 | 0.0096 | 0.0049 | 0.0052 | 0.0173 | 0.0454 | 0.0042 | 0.0027 |
| 8 (16) | 0.0856 | 0.0560 | 0.0376 | 0.0058 | 0.0540 | 0.1192 | 0.2595 | 0.0043 | 0.0002 |
| 4 (32) | 0.1602 | 0.0925 | 0.0596 | 0.0058 | 0.0920 | 0.1970 | 0.4239 | 0.0042 | 0.0016 |

Two facts follow. (a) A spread overstated 2× is worse than no spread. A spread overstated 1.5× keeps only about a third of the gain. (b) The curve is not monotone in the "truth". A variance floor of 0.1² beats the true std by 30× at 8 calls, because the extra spread cancels the Euler error of the sub-steps. A better FD with a modified table therefore does not show that the table is closer to the posterior.

**How to read a null result.** Suppose `gm_ode_cal` does not beat the point model's Euler at 8 calls by 5%. That does not show that the learned distribution is needed. The table can be too large (model bias, and F2) or too coarse: one value per t, while K = 1 learns s_u per sample. A null result is decisive only for the claim "the point model's own error, used as the spread, does not transfer the gain." Suppose `gm_ode_cal` matches K = 1's GM-ODE gain (−3.8% / −6.1% at 8 calls). Then the sub-step gain does not need a learned distribution, and the claim is strong, because the table is a conservative choice.

**Scaled tables (0.5×, 0.7×) are a tuning knob, not a fair follow-up, if they are chosen on test FD.** The toy shows a non-monotone response, so picking the best scale on the 50K test FD is selection on the test set. Two fair options exist. (1) Fix the arms now, before any result, and report all of them: raw table, bound-clipped table (F2), and if wanted 0.5× and 0.7×. Apply the 5% decision line to one arm that you name in advance, for example the bound-clipped table. (2) Choose the scale on the val split (`data/gpic/gpic_val_50k.jsonl` captions and `reference_stats/val_stats.npz`), then run one test evaluation with the chosen scale. F1 must be fixed first in either case, or the second table silently reuses the first result.

## Findings

| # | Severity | file:line | Failure scenario | Fix |
|---|---|---|---|---|
| F1 | HIGH | jobs/eval_gmflow.sh:56-57, jobs/evals_gmflow.sh:45 | A second `gm_ode_cal` table (scaled, clipped, or the other step's) for the same step and calls is skipped as "done before" with exit 0. The old FD is then read as the new arm. Also, the 033408 table can be used on step 066816 without an error. | Assert that the table's `ckpt` matches `ms-step=$STEP`. Add a table tag (hash or `tag` field) to the output name and to the evals done-check. |
| F2 | HIGH (50 calls) / MEDIUM (≤ 16 calls) | tools/gmflow/calibrate_spread.py:136-139; gm_spread.py:23-29 | For t > about 0.94 the table (smoke: 0.055 at t = 0.99) exceeds the provable posterior-std bound (1 − t)/t (0.0101). The 50-call arm runs its last 3 calls with a spread up to 5× too large. The toy shows that 2× already loses to the exact step. | Use `min(s_x, (1 − t)/t)`: a column in `merge`, or a second arm fixed now. Report both. |
| F3 | HIGH (procedure) | docs/gmflow_hn_impl_report_2026-11-05.md §5 item 1; docs/gmflow_hn.md | "Scaling the table (0.5×, 0.7×) is the natural follow-up." If the scale is picked on test FD, the result is tuned on the test set. The toy shows a non-monotone response (a variance floor beats the true std). | Before the run, fix the list of table arms and the one arm that the 5% line judges. Or choose the scale on the val split, then run one test eval. |
| F4 | MEDIUM | tools/gmflow/bench_sampling.py:121-131, :150-158 | The bf16 end-to-end check (mean < 5e-3) can fail without a bug, or pass with a small kernel error. The per-call fp32 check covers only the first call (σ = 1), not the last call (σ → 0), where the clamps act. | Add a per-call check at t = 1 − 1/calls. Make the end-to-end check fp32 with tolerance 1e-4. Keep the bf16 numbers as a report only. |
| F5 | MEDIUM | baselines/PixelGen/src/diffusion/flow_matching/gmflow.py:80; gm_ode_triton.py:89, :135 | With `EVAL_SUBSTEPS` such that calls × n ≥ about 1000, and s_x² + σ² < 1e-6 at the last sub-steps, the clamp shrinks the means. With s_x = 0: rms 3.5e-2 at 8 × 128 and 0.46 at 8 × 192 against the exact step. | Lower the clamp (1e-20), or keep the means unchanged where v = 0. Until then, reject CALLS × EVAL_SUBSTEPS > 512. Add a T22 case at calls × n = 1024. |
| F6 | MEDIUM | jobs/evals_gmflow.sh:25 | `EVALS_WT` cannot reach a wq job (clean environment), so a gm_ode_cal list cannot be queued without a wrapper. It also conflicts with gmflow's `--wt`. | At the merge, use gmflow's `--wt <worktree>` and drop `EVALS_WT`. Until the merge, use a wrapper that exports it inside the script. |
| F7 | MEDIUM | merge: sampling_gmflow.py:76-99, jobs/eval_gmflow.sh:34, :74-75, jobs/evals_gmflow.sh:14-25, tools/gmflow/mutate.py:19, tools/gmflow/test_gmflow.py:422 | Five conflicts (section 5). After a naive resolution, `order=2` with a table runs an untested sampler, and `EVAL_SUBSTEPS` rejects `gm_ode2`. | Resolve as in the table in section 5. Assert not (table and order 2). Extend `EVAL_SUBSTEPS` to `gm_ode2`. Port T27 to `--wt`. Add fused tests on a per-pixel and an order-2 mixture. |
| F8 | LOW | jobs/eval_gmflow.sh:45, :59, :75 | A relative `EVAL_SPREAD_TABLE` passes the check, then `cp` and `main.py` fail after `cd` (a loud failure). | `realpath` the table in both scripts. |
| F9 | LOW | jobs/calibrate_spread.sh:188 | `commit.txt` at the root is overwritten per run. If the two tables come from different starts, the record is wrong. | Write `commit.txt` into each `<step>/` directory, next to `table.json`. |
| F10 | LOW | tools/gmflow/calibrate_spread.py:58, :71, :130 | `noise_for` uses `len(T_GRID)`, not `len(ts)`. `merge` does not compare `images` across shards. Production is not affected. | Pass `len(ts)` to `noise_for`. Add `images` to the merge assert. |
| F11 | LOW | sampling_gmflow.py:121; bench_sampling.py:57-59 | n + 1 small device ops per call build the σ grid (65 at n = 64), which dilutes the fused speedup. `init_process_group` runs before `set_device`. | Build the grid with one op, or on the host. Call `set_device` first. |
| P | precondition | jobs/calibrate_spread.sh:182, jobs/bench_sampling.sh:188 | `gpic-wt/gmflow_hn_run` does not exist, so both jobs exit 2 with "holds old code". | Create the frozen worktree at the reviewed commit (or at the fix commit) before submitting. |

Confirmed claims of the implementation report: 17 tests pass. 9 mutants are killed. T22 differences are 2.4e-7 to 4.2e-7. The defaults are unchanged (now confirmed bitwise against c16d71d, which is stronger than the report states). The kernel equals the reference, both s_x shapes are supported, and the held-out data is clean. jsonargparse accepts the new options (report §4 item 4, now tested at the CLI layer). I contest one claim: report §5 item 1 ("scaling is the natural follow-up"), see F3.

Per job:
- `jobs/calibrate_spread.sh`: GO, after P. The tables are needed by every later step. The F2 fix can be applied in `merge` afterwards, because the shard sums do not change.
- `jobs/bench_sampling.sh`: GO, after P. F4 weakens only the correctness check, not the throughput numbers, and the JSON is written even when a check fails.
- `gm_ode_cal` evaluations: NO-GO until F1 is fixed, the table arms are fixed in advance (F2, F3), and the job can reach the frozen worktree (F6).

VERDICT: NO-GO
