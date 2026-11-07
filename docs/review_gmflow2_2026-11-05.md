# Review of group D (per-pixel spread, GM-SDE 2 / GM-ODE 2), branch `gmflow`, commits c16d71d..251b3fa

Reviewer: an independent, adversarial pass (2026-11-06). The code was not changed. All Python ran on vis1 / vis2 (CPU). Nothing was submitted, cancelled or pushed.
Reviewer scripts and raw outputs (outside the repo): `/scratch/gpfs/GROUP/USER/tmp/review_gmflow2/` (`check_order2.py`, `check_clamp.py`, `diag_t14.py`, `parser_test.py`, `dry_train.sh`, `*_out.txt`, `tests_*.txt`). Merge clone: `/scratch/gpfs/GROUP/USER/tmp/review_merge/repo` (uncommitted merge).

## Summary

1. The math is right. Our GM-SDE 2 and GM-ODE 2 match my own transcription of the official `forward_test` + `gm_2nd_order` end to end. I used an imperfect network, so the correction is not zero. GM-ODE 2 agrees to 3e-6 – 6e-6 at 4 and 8 calls; GM-SDE 2 agrees to 2.4e-7. Order 1 differs from both by about 2. The per-pixel spread lands on the right pixel. It is shared over K and C, and the loss treats it per pixel. The s-MLP still gets its gradient, and the trunk gets none through the head.
2. `gm_spread image` and `order 1` stay bitwise unchanged (T7, T11, T14 pass). r and STEPS are correct.
3. No BLOCKING finding for the smoke. Two HIGH findings matter for how the real results are read, and one for the merge:
   - H1: at 50 calls, order 2 is exactly order 1 on any realistic network. At 16 calls it is order 1 in practice for high-variance images. The cause is the official clamp (`err_power / prev_h^2 ≥ 1`), which we copy faithfully. This holds for every evaluation at ≥ 32 calls.
   - H2: the implementer's stated deviation from the official code (no 1e-6 clamp on `var − v'`) is not always negligible. With a small bias at a small variance, it reaches 1.5 – 4 levels of 255 in my test. It does not invalidate the method, but the report's "5e-6" understates it.
   - H3 (merge): merged with `gmflow-hn`, T27 fails until it uses `--wt` instead of `EVALS_WT`. On vis2 alone, T14 fails on a `torch.cdist` rounding artifact; the same code passes on vis1.
4. VERDICT at the end: GO for the smoke; GO for the real runs after the smoke passes, with the conditions listed there.

## 1. Per-pixel spread

| Item | Evidence | Result |
|---|---|---|
| Placement | `JiT_T2I.py:563-566`: `h = final_layer.norm_final(x)`, `x = final_layer.linear(h)`, `spread = unpatchify(gm_spread_head(h.detach()))`. `FinalLayer` (`JiT_T2I.py:226-238`) has no adaLN, so `norm_final(x)` is exactly the input of the final linear. | correct |
| Detach | `h.detach()` at `:566`. My check: with a loss on `logstd` alone, the trunk gradient is 0.0, the head gradient is 7.4e4 and the s-MLP gradient is 5.5e3 (K = 8; K = 1 similar). | correct |
| Zero init, RNG | `:400-404`: built after `initialize_weights()` inside `fork_rng(devices=[])`; weight and bias set to 0. T11 checks parameters, the RNG stream and outputs. | correct |
| Unpatchify alignment | My test (`check_order2.py` C): head bias = r·16 + c for each in-patch position, and the same pattern in the bias of the means' channel (k = 0, c = 0). Both land at pixel (r mod 16, c mod 16) of every patch, at K = 8 and K = 1. The offset and the means use the same `fold`. | correct |
| Shape / broadcast | `logstd + spread` = (B,1,1,1,1) + (B,1,1,H,W) → (B,1,1,H,W) (`:575-576`): broadcast over K (dim 1) and C (dim 2). | correct |
| Mean mode | `gm_mode="mean"` takes the old branch (`:567-568`). It equals the image model with the same weights, bitwise (my check). | correct |
| Loss | `gm_nll` (`gmflow.py:37-44`): `inverse_std` clamp and `- logstd` broadcast to (B,K,1,H,W), `.sum(dim=2)` over C. My check: a per-pixel logstd gives exactly the per-pixel choice of two scalar-logstd losses. d nll / d log s = 3.0 at zero residual with λ = 1, so the D log s term counts the 3 channels per pixel. `transition_nll` (`:47-54`): `c2^2 s_x^2 + c3` per pixel through `logaddexp`. | correct |
| s-MLP gradient | Gets gradient through `logstd + spread` (above). | correct |
| image bitwise | T11 (fp32 and bf16, K = 8 and 1, outputs and all gradients against `git show c16d71d`) PASS on vis1 and vis2. | correct |
| Diagnostics | `training_gmflow.py:67-73`: `gm/s_u` is now the per-sample mean of s_u over pixels (for image it is the old value: one element). | correct |

Test gap (LOW, L1): no test fails if the offset is spatially permuted. I planted `spread.transpose(-1, -2)` (offset of pixel (r, c) placed at (c, r)). T9, T11, T12, T13 and T14 all PASS (a temp copy of `src` at `tmp/review_gmflow2/mut`, the five tests run against it on vis1). The T11 perturbation test only checks that the std over pixels is > 0, and the toy tests (T12, T13) pass logstd straight to the loss and the samplers, never through the model. The code is right today (my alignment test above); the suite would not catch a later break. Fix: add my alignment check (the bias pattern r·16 + c against the means' unpatchify) to T11.

## 2. Second order

### 2.1 Line by line against the official code

| Official (gmflow.py) | Ours | Same? |
|---|---|---|
| `:259-264`: `gm_mean = gaussian_output['mean']`; `base var = var.mean((-1, -2))`; `avg_var = mean((-3,-2,-1))` | `gm_iso_var` (`gmflow.py:97-105`): per pixel `Σ A_k |μ_k − mean|² / C + s_x²`, then mean over H, W; `avg_var` `:126` | yes. With a per-pixel s_x, `gm_vars` (B,1,1,H,W) enters `gm_to_iso_gaussian_jit` per pixel before the H, W mean. That is what the official code would do with a per-pixel `logstds` (its `gm_to_iso_gaussian` docstring allows (bs,1,1,h,w)). |
| `:266-268` `mean_from_prev = denoising_gm_convert_to_mean(prev_gm, x_t, prev_x_t, t, prev_t)` (Eq. 10) | `sampling_gmflow.py:46-56` `gm_convert_mean(p_gm, p_sigma, sigma, p_x, x)` | yes: same argument roles (src = previous, tgt = current); `gm_convert` (`gmflow.py:73-86`) = `denoising_gm_convert_to_mean_jit` term for term |
| `:267` `self.prev_gm = gm_output` (raw, before the correction) | `state = (gm, …)` stored before `gm_second_order` (`:49-50`) | yes |
| `:271` `k = 0.5 * h / prev_h` | `gmflow.py:127` | yes |
| `:272` `prev_h_norm = prev_h / num_timesteps` | `prev_h` is already in sigma units | yes |
| `:274` `err_power = avg_var * (g² c_b + c_a)`, g = 0 | `:128` `avg_var * ca`, `GM2_CA = 0.005` (`:94`) | yes |
| `:275-276` scale `sqrt(clamp(1 − err/prev_h², 0)) · k` | `:129` | yes |
| `:283-284` `bias_power` over (C,H,W); cap `sqrt(min(1, avg_var / max(bp, 1e-6)))` | `:131-132` | yes |
| `:286-288` `var · clamp(1 − bp / max(avg_var, 1e-6), 1e-6)`, bp before the cap | `:133-134` | yes |
| `:289-291` `gm_mul_iso_gaussian(base_gm, iso_gaussian_mul_iso_gaussian(out, base, 1, −1), 1, 1)` | closed form `:135-142` | yes where the two 1e-6 clamps are inactive; see H2 |
| `:312-313` first call: `prev_gm = gm_output`, no change | `:51-52` `if prev is None: return gm, state` | yes |
| `:315-317` state `prev_x_t, prev_t, prev_h` | `state = (gm, x, sigma, h)` | yes |
| `init_gm_cache()` once per `forward_test` | `prev = None` local to each `_impl_sampling` call (`:71`, `:107`) | yes: reset for every batch; no state on the object, so nothing leaks between batches, prompts or ranks |
| `:505-509` h = t − t_next of the call (next call boundary, not the sub-step), last call h = t | `h = sigma − sigma_to` with `sigma_to` the next call's sigma; last call `sigma_to = 0` → h = sigma | yes |
| `:527-539` sub-steps convert the **corrected** `gm_output` with `x_t_base` = the call's state | `GMODESampler` loop converts the corrected `gm` with `x_src = x` (`:115-122`) | yes |
| `gm_to_model_output(…, 'sample')` + `GMFlowSDEScheduler.step` (small variance) | `gm_draw` + `bridge_coefs` | yes (no spectral sampling: unchanged from order 1) |
| `gm2_correction_steps` default 0 | not ported | fine |

### 2.2 An end-to-end test with a nonzero correction (question 2)

`check_order2.py` part A. I wrote my own transcription of `forward_test` (no guidance, order 2). It covers the scheduler grid `1 − linspace(0, 1, N·n_sub, endpoint=False)`, `FlowEulerODEScheduler.step` and `GMFlowSDEScheduler.step` (the draw uses our `gm_draw` with the same RNG order), `gm_2nd_order`, `gm_to_iso_gaussian_jit`, `denoising_gm_convert_to_mean_jit`, `iso_gaussian_mul_iso_gaussian_jit` and `gm_mul_iso_gaussian_jit`, all with the official clamps. It runs in float64. The network is a fixed nonlinear GM that is not the posterior of anything: x-dependent means, weights and logstd, and an x-dependent per-pixel offset for "pixel".

| | calls (sub-steps) | max \|ours₂ − official₂\| | max \|ours₁ − official₂\| | max bias_power / avg_var | calls with scale clamped to 0 |
|---|---|---:|---:|---:|---:|
| GM-ODE 2 image | 4 (32) | 3.4e-6 | 1.87 | 0.092 | 0 |
| GM-ODE 2 image | 8 (16) | 5.8e-6 | 2.11 | 0.092 | 0 |
| GM-ODE 2 pixel | 4 (32) | 7.1e-7 | 2.02 | 0.092 | 0 |
| GM-ODE 2 pixel | 8 (16) | 6.2e-6 | 2.75 | 0.089 | 0 |
| GM-SDE 2 image / pixel | 4, 8 | 2.4e-7 | 2.4 – 2.95 | | |
| GM-ODE 2 image / pixel | 50 (3) | 6.9e-4 / 5.2e-4 | 6.9e-4 / 5.1e-4 | 0.000 | 49 of 49 |
| GM-SDE 2 image / pixel | 50 | 4.8e-7 / 2.4e-7 | 2.4e-7 / 1.2e-7 | | |

At 4 and 8 calls the correction is large (order 1 is 2 away), and ours equals the official algorithm to fp32 rounding (the network runs in fp32). At 50 calls my script flags a "failure" because order 2 does not act: both ours and the official code are order 1 there, and the 5e-4 – 7e-4 gap is only the fp64 (official) vs fp32 (ours) arithmetic over 150 sub-steps. This is finding H1.

### 2.3 H1: order 2 is a no-op at 50 calls (and often at 16)

The scale is `sqrt(max(0, 1 − c_a avg_var / prev_h²))` (`gmflow.py:129`, official `:275-276`). On our uniform grid prev_h = 1/N, so the scale is 0 whenever `avg_var ≥ prev_h² / 0.005 = 200 / N²` (x-space variance of the isotropic surrogate, data in [−1, 1]):

| calls | the correction is 0 when avg_var ≥ |
|---:|---:|
| 4 | 12.5 |
| 8 | 3.125 |
| 16 | 0.781 |
| 32 | 0.195 |
| 50 | 0.080 |

Early calls have avg_var ≈ data variance (≈ 0.2 – 0.3 for images in [−1, 1]); late calls have small avg_var. At 50 calls the correction vanishes on every call where avg_var ≥ 0.08; in my toy, that was all 49 calls. At 8 calls it never vanishes for real data. Consequences:
1. "GM-ODE 2 at 50 calls" will be FD-identical to GM-ODE at 50 calls up to the late low-variance calls. Do not queue 50-call order-2 evaluations expecting a difference; if you do, read a tie as "the clamp", not "order 2 does not help".
2. This matches the official code (and the paper's `max(0, ·)`), so it is not a bug. It is missing from the implementer's report and from `docs/gmflow.md` section 7.
3. The judgement line in `docs/gmflow.md` 6.2 item 4 ("second order vs first, 5% at C0/16 and C0/8") is fine at 8 and 4 calls. At ≥ 32 calls the comparison is degenerate.

### 2.4 H2: the stated deviation (no 1e-6 clamp on var − v') can be visible

The report (`gmflow2_impl_report_2026-11-05.md:97`, `:199`) says the official clamp matters only when the correction is 0, and then by about 5e-6. That holds at var ≈ 0.3 (T14's random mixtures). Without guidance var = avg_var, so `var − v' = bias_power` (before the cap). The official clamp `max(var − v', 1e-6)` therefore acts whenever bias_power < 1e-6: an absolute threshold, an RMS bias under 1e-3 (an eighth of a level of 255). There r_var ≈ var² / 1e-6, and each component mean is pulled toward 0 by a fraction ≈ s_x² · 1e-6 / var². That fraction grows as var shrinks. `check_clamp.py` part D, one call at sigma 0.25, means in [−0.5, 0.5], ours vs the official product, max difference of the mixture mean in 1/255 levels:

| var | bias_power | clamp active | ours moves the mean by | ours − official |
|---:|---:|---|---:|---:|
| 1.5e-2 | 3.5e-7 | yes | 0.27 | 0.005 |
| 1.2e-3 | 3.6e-7 | yes | 0.27 | 0.085 |
| 8.1e-5 | 3.8e-7 | yes | 0.41 | 1.48 |
| 1.3e-5 | 3.5e-7 | yes | 0.27 | 3.96 |
| any | ≥ 4e-6 | no | 0.8 – 29 | 0.0000 |

1. Where the clamp is inactive, ours equals the official code exactly (also T14, 1e-15).
2. Where it is active (small bias at small variance: late calls, smooth regions), the official code pulls every component mean toward 0 by up to 4 levels per call. Ours keeps the exact ratio, which is the paper's formula. This is an artifact of the official code, so ours is the better choice. But "follows the official code" is then not exactly true. Late calls of GM-ODE 2 on a trained model (var ~ 1e-4 – 1e-5 at sigma ≤ 0.25) are the regime where it acts.
3. Severity HIGH for the claim and the docs, not for the GPU jobs: the FD of "GM-ODE 2" is ours, not GMFlow's, in that regime. Fix: say so in `docs/gmflow.md` 7.2 item 6 and in the report (replace "5e-6" with "up to about 4/255 per call at var ~ 1e-5 with bias_power < 1e-6"). Optionally add a T14 case at var ~ 1e-5. If a paper claims "GMFlow's GM-ODE 2", either switch to the clamped form or state the deviation.

### 2.5 Other second-order points

1. **The first-call fallback**: correct (`sampling_gmflow.py:51-52`; mutant `no_first_order_fallback` killed by T14).
2. **The per-pixel spread inside avg_var**: an image average of `Σ A_k|μ_k − mean|²/C + s_x²` (`gm_iso_var`). This equals what the official code does with per-pixel logstds. With a per-pixel spread, the cap and `err_power` are set by the high-variance pixels (report risk 2). This is acceptable; it is the paper's isotropic surrogate.
3. **The point-model path**: GM-SDE skips order 2 for a tensor output (`:76`). GM-ODE applies it to `as_mixture` (K = 1, s = −inf) and the `var ≤ 1e-12` guard returns it unchanged (`gmflow.py:143`). `eval_gmflow.sh:40` refuses `0:gm_sde2|0:gm_ode2` anyway.
4. **The `var ≤ 1e-12` guard** (`gmflow.py:143`): per image, not per pixel. It acts only when the whole image is certain. Fine.
5. **`h`, `prev_h` as tensors**: `h = sigma − sigma_to` is a (1,1,1,1) tensor, so `k` and `scale` stay tensors and there is no host sync. Fine.
6. **Mutants** (re-run by me on vis1, `mutants_out.txt`): all five new mutants are killed (section 6).

## 3. FLOPs and STEPS

1. The hand count of the head: Linear(640, 256) per token, 256 tokens → 2 · 256 · 640 · 256 = 83,886,080 FLOP = 83.89 MFLOP. T9 measures exactly this difference with `FlopCounterMode` (`test_gmflow.py:374-397`): pixel − image = 2517.90 − 2434.01 = 83.89 MFLOP at K = 8 and 169.08 − 85.20 = 83.88 at K = 1.
2. The denominator 34.72 GFLOP = `ladder_runs.json` d640 `gflop_per_img` 104.2 / 3 = 34.733; T9 uses 34.72 (`:381`), as section 6.1 item 1 does for the existing runs (r = 0.07010 there). It is consistent across arms; the 0.04% difference in the denominator cannot move a STEPS by 128.
3. K = 8 pixel: r = 2517.90 / 34720 = 0.072520; 62208 · 1.072520 = 66719 ≤ 66816; 62336 · 1.072520 = 66857 > 66816. **STEPS = 62208**, correct.
4. K = 1 pixel: r = 169.08 / 34720 = 0.0048698; 66432 · 1.0048698 = 66756 ≤ 66816; 66560 · 1.0048698 = 66884 > 66816. **STEPS = 66432**, correct.
5. MS = STEPS / 4 (`train_ladder.sh`): 15552 and 16608. C0/16 = 31104 and 33216. These are the checkpoints the evaluations must name.
6. Note (LOW, L2): `docs/gmflow.md:81` still says MS = 15584 / 16640 for the image runs (62336 / 4 = 15584 is right; 66560 / 4 = 16640 is right); no error, only a reminder that the pixel runs' MS differ (15552 / 16608).

## 4. The job interface

### 4.1 Existing calls and names (diff c16d71d..HEAD)

1. `eval_gmflow.sh`: the existing samplers keep their `SARGS` lines byte for byte (`:36-39`, `:42`). The output directory stays `eval_<step>_<sampler>_calls<calls>` (`:51`). For image, `SPREAD_ARGS` is empty (`:72`), so the `main.py predict` command line equals the old one. `ckpt.txt` gains a `spread=` field (`:71`). Nothing parses it (grep over `jobs/`, `tools/`), so this is fine. The old K = 0 check moved into a `case` with the same refusals plus the two new samplers (`:40`).
2. The guard (`:68`): the checkpoint must hold `ema_denoiser.gm_spread_head.weight` iff `EVAL_SPREAD = pixel`. A pixel checkpoint loaded as image (`strict=False`) would otherwise silently drop the head. A K = 0 run with pixel is refused before the guard (`:32`).
3. `evals_gmflow.sh`: `--link`, then `--wt`, then `--spread` (`:22-27`), in that fixed order. The default worktree is `gmflow_run2` and the default spread is image. It exports `EVAL_SPREAD` (`:44`).
4. `train_gmflow.sh`: `--wt`, then `--spread` after `<data>` (`:26-29`). The pixel run passes `--model.denoiser.init_args.gm_spread pixel` (`:40-43`, `:59`), records `spread.txt` and refuses another spread (`:50-53`). It forwards `--wt` / `--spread` to the evaluations (`:65-66`).
5. **My dry run** (`dry_train.sh`, stub worktrees and stub sub-scripts, old c16d71d script vs HEAD script, on vis1):
   - `8 d640 640 9 2 62336 t1 w1147`, `… t2 w1147 31168:gm_ode:8` and `1 d1024 1024 12 3 160512 t3 ladder 1000:gm_ode:8 2000:euler_mean:50`: identical `train_ladder.sh` and `evals_gmflow.sh` arguments at c16d71d and at HEAD.
   - `… --wt newwt --spread pixel 31104:gm_ode2:8`: the training gets `gm_spread pixel`; the evaluations get `--wt newwt --spread pixel`.
   - The same tag restarted without `--spread`: refused (`started with spread pixel, not image`).
   - `--spread pixel` on `gmflow_run2`: refused. One cosmetic issue: grep prints a "No such file" line first, because the stub had no `JiT_T2I.py`; the real `gmflow_run2` has one.
   - At c16d71d, `--wt newwt --spread pixel` would be silently taken as **task names** and the run trained as image. This only matters if someone points a wq job at an old checkout of the script. The scripts run from `$WT/jobs/…` of the chosen worktree, so use the new worktree's copy.
6. **`--wt` is a name, not a path** (`$WC/project/gpic-wt/$WTN`). There is no check that the name has no `/` or `..`. LOW.

### 4.2 Fetchable rules

1. All the scripts keep `#FETCHABLE`, `--comment=fetchable` (eval, evals, smoke) or `fetchable-partial` (train), `-c 96`, `--gres=gpu:8` and `source …/env.sh`. env.sh sets the per-job TMPDIR and `proxy/default`.
2. Self-contained: worktree and spread come from arguments (`--wt`, `--spread`), not from the submit shell. `EVAL_SPREAD`, `GMFLOW_WT` and `LADDER_RUN_ROOT` are set by `evals_gmflow.sh` inside the job. The smoke sets every variable it needs in its own body. A direct `wq submit eval_gmflow.sh …` cannot set `EVAL_SPREAD` (it would default to image, and the guard would then refuse a pixel checkpoint loudly, not silently). Pixel evaluations must go through `evals_gmflow.sh --wt … --spread pixel` or `train_gmflow.sh`. MEDIUM (M3) as a usage trap; the guard makes it fail closed.
3. `eval_gmflow.sh` still defaults `WT` to `gpic-wt/gmflow` (the live development worktree, `:24`) when `GMFLOW_WT` is unset. This predates group D. A direct submission evaluates whatever the dev worktree holds at run time. MEDIUM (M4), pre-existing.

### 4.3 jobs/smoke_gmflow2.sh

| Required | Where | Covered? |
|---|---|---|
| compile, DDP | `train_ladder.sh` (8 GPUs, `net.compile()` in `configure_model`) | yes |
| pixel model trains | `TRAIN` with `gm_spread pixel` (`:63`); `check` requires `logsu_pix_std` > 1e-4 (`:43`) | yes |
| ms checkpoint | 1600 total → MS = 400 (`:65-68`) | yes |
| resume from last.ckpt | last every 50 → resume to 500; grep `resuming from .*last` (`:71-74`) | yes |
| predict gm_ode, gm_ode2, gm_sde2 on the pixel model | `:77-81`, `EVAL_SPREAD=pixel`, 8 calls, 512 prompts | yes |
| the guard | pixel checkpoint as image must fail (`:83`) | yes; it reports only the first matching line, and any non-zero exit (for example a timeout) counts as "refused". LOW |
| w1173 gm_ode / gm_ode2 / gm_sde2 + paired diff | `:85-94`, link into the smoke dir, nothing written in w1173's dir | yes |
| number-pattern parser | `F` (`:36`); my test on run-together lines (three rank lines glued, a `nan`, a `1.2e-05`): 4 rows, the right fields, `nan` kept and failed by `isfinite` (`parser_test.py`) | yes |
| recompiles | counts `__recompiles` after `[warmup] done` (`:44`, `:50`); predict logs grep without qwen3 (`:96`) | yes; `check` counts them but does not fail on them. LOW |
| under 50 min | timeouts 480 + 300 + 3·240 + 180 + 3·240 = 2400 s = 40 min, plus link and checks. The first smoke (w1168) took 5 min to train 800 steps and 75 – 110 s per predict. 46.5 min worst case under a 50 min wall | yes, but tight: one slow node and the w1173 gm_sde2 predict is cut. A cut predict is `rc ≠ 0` → `fail=1`, which shows up. LOW |
| w1174 | not in the smoke; w1174 only differs in K (1) | acceptable: K = 1 order-2 code path is the same; T14 covers K = 1 shapes via random GMs |

The smoke does not test a K = 1 pixel model, but T11 and T9 cover K = 1 on CPU. The w1173 gm_sde2 paired diff is only computed for gm_ode vs gm_ode2 (`:93`); fine.

One smoke gap (MEDIUM, M1): the smoke evaluates order 2 only at 8 calls. It cannot show H1. It also cannot show whether `gm_ode2` at 4 calls (32 sub-steps) is stable on w1173. Add `gm_ode2 4` on w1173 if the wall allows. It does not, at 46.5 min, unless one pixel predict is dropped: `gm_ode` on the pixel model is redundant with the order-2 runs for a smoke.

## 5. Merge with gmflow-hn

Temporary clone: `/scratch/gpfs/GROUP/USER/tmp/review_merge/repo`. `git merge --no-ff --no-commit origin/gmflow-hn` into `gmflow` (251b3fa), against ffa4bed. Neither branch was changed. The merge was not committed.

Conflicts (5 files) and the resolutions I applied in the clone:

| File | Conflict | Resolution |
|---|---|---|
| `baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py` | `GMODESampler.__init__` signature + docstring | gmflow's docstring (order 2) + hn's signature (`spread_table`, `fused`). The loop body auto-merged: `_second_order` runs before `fused` / the sub-step loop, so `fused` gets the corrected mixture. Correct order. |
| `jobs/eval_gmflow.sh` | the `ckpt.txt` line and the next line | gmflow's `ckpt.txt` line (with `spread=`), then hn's `spread_table` line, then gmflow's `SPREAD_ARGS`. Everything else auto-merged (both sampler lists, `EVAL_SUBSTEPS` + `_n<n>`, the guard). |
| `jobs/evals_gmflow.sh` | usage comment; option parsing | order `--link`, `--wt`, `--spread`, `--table`; **drop `EVALS_WT`** and keep `WT=$WC/project/gpic-wt/$WTN`. The 4-field task loop auto-merged. |
| `tools/gmflow/mutate.py` | the mutant dict | keep both lists (5 + 4) |
| `tools/gmflow/test_gmflow.py` | T11–T14 vs T21–T27 | keep both blocks. No duplicate names. |

Further merge issues the conflict list does not show:

1. **`EVALS_WT` callers.** `docs/gmflow_hn.md:59, 95, 104` and `docs/gmflow_hn_impl_report_2026-11-05.md:17, 57, 167` tell the reader to set `EVALS_WT`. After the merge that variable does nothing. Fix: rewrite them to use `--wt gmflow_hn_run`. This also removes hn's own risk 6, because `--wt` is an argument, not environment.
2. **T27 depends on `EVALS_WT`** (`test_gmflow.py`, hn's T27). The stub worktree is passed through `EVALS_WT=<path>`. After the merge, T27 fails with "…/gmflow_run2 holds old code". Fix (applied in the clone): build the stub at `$tmp/project/gpic-wt/stubwt` and insert `--wt stubwt` after the 6 positional arguments of each `evs` call.
3. **`--wt` + `EVAL_SUBSTEPS`**: `EVAL_SUBSTEPS` is refused for `gm_ode2` (`case $SAMPLER in gm_ode|gm_ode_cal`). That is consistent but limits a later sub-step sweep of GM-ODE 2. LOW.
4. **The fused kernel after order 2**: `_fused_substeps` reads the corrected `gm["logstd"]`. In `gm_second_order`, `denom = v5 + g · beta5` with v5, beta5 (B,1,1,1,1) and g = s_x² of the input's shape. So the corrected logstd stays (B,1,1,1,1) for image and (B,1,1,H,W) for pixel. hn's `_check` accepts both shapes, and its stride choice `(1, 0)` vs `(HW, 1)` handles both. T25 and T26 test per-pixel s_x. OK. Not tested: `fused=True` together with `order=2`. No current job uses that combination.
5. **The default worktree of eval_gmflow.sh** stays `GMFLOW_WT` or `gpic-wt/gmflow`. Unchanged by the merge.

Merged test suite (`tools/gmflow/run_vis.sh tools/gmflow/test_gmflow.py`):

| node | resolution | result |
|---|---|---|
| vis2 (AMD EPYC 9454) | conflicts only | T14 FAIL, T27 FAIL, the rest PASS |
| vis1 (Xeon 8380) | conflicts + T27 fix | **T1–T14, T21–T27 ALL PASS** (`tests_merged_vis1.txt`) |
| vis1 | T14 + T27 alone | PASS |

**The T14 failure on vis2 is not the merge** (H3). It fails the same way on the unmerged `gmflow` HEAD on vis2, and passes on vis1 for both trees (`t14_iso.txt`). `diag_t14.py`: on both nodes the second-order bias is ≤ 1e-13 of avg_var (exact posterior, zero correction), and the samples differ only by fp32 rounding (−0.8000000715 vs −0.800000012). The landing metric uses `torch.cdist` (`test_gmflow.py:591`), whose matrix-multiply path on the AMD node returns 3.45e-4 for a true distance of 6.1e-8 (`check_clamp.py` E: exact 6.14e-8 on both nodes; cdist 0 on vis1, 3.45e-4 on vis2). T14 then asserts order 2's `dist0` ≤ order 1's + 1e-4. Fix: `torch.cdist(v, P, compute_mode="donot_use_mm_for_euclid_dist")` or an explicit `(v[:, None] - P).norm(dim=-1)` in `landing` (`:591`) and `toy_stats` (`:217`). The run_vis.sh instructions in the brief name vis1, so the implementer never saw it.

## 6. Tests on the vis nodes

1. `ssh vis1 'bash …/gmflow/tools/gmflow/run_vis.sh tools/gmflow/test_gmflow.py'` on HEAD 251b3fa: **T1–T14 ALL PASS**, exit code 0 (`tests_head.txt`). T9 numbers as in section 3; T11 as in the report.
2. The same on vis2: T14 FAIL (H3, cdist on AMD); the rest PASS.
3. Mutants (the 5 new ones, re-run on vis1, `mutants_out.txt`): all 5 are killed. `spread_input_not_detached` → T11; `spread_varies_over_channels` → T11, T14; `avg_var_wrong_axis` → T14; `k_without_half` → T14; `no_first_order_fallback` → T14. The planted transposed offset (L1) survives.
4. My own checks (`check_order2.py`, vis1): part A as in 2.2; part C all ok (11 checks: alignment, sharing, mean mode, gradients, per-pixel loss, D log s).

## Findings

| # | Severity | Where | Failure scenario | Fix |
|---|---|---|---|---|
| H1 | HIGH | `gmflow.py:129` (as official `:275-276`); `docs/gmflow.md` 7; report | At ≥ 32 calls (all 49 calls at 50 calls in my test) `1 − 0.005 avg_var / prev_h² ≤ 0`, so GM-ODE 2 = GM-ODE and GM-SDE 2 = GM-SDE. A 50-call order-2 evaluation shows a tie that is read as "order 2 does not help". | Do not queue order-2 evaluations at ≥ 32 calls, or label them as clamp-dominated. Add the table of section 2.3 to section 7 and the report. |
| H2 | HIGH (claim / docs) | `gmflow.py:135-142`; report `:97`, `:199`; `docs/gmflow.md` 7.2 item 6 | With bias_power < 1e-6 at var ~ 1e-4 – 1e-5 (late calls), the official product moves the mean by up to 4/255 per call and ours by 0. "Follows the official code" and "5e-6" are wrong there. | Correct the docs (state the size). Keep the exact form for our runs; state the deviation in any write-up that names GMFlow's GM-ODE 2. Optionally add a T14 case at var ~ 1e-5. |
| H3 | HIGH (merge / test) | `test_gmflow.py:591` (`landing`), `:217` (`toy_stats`); hn's T27 | (a) On vis2 (AMD), `torch.cdist` returns 3.45e-4 for a distance of 6e-8, so T14 fails on correct code. (b) After the merge, T27 fails because `EVALS_WT` is gone. | (a) `compute_mode="donot_use_mm_for_euclid_dist"` or an explicit norm. (b) T27 with `--wt stubwt` (the patch is in the merge clone). |
| M1 | MEDIUM | `jobs/smoke_gmflow2.sh:77-94` | The smoke checks order 2 only at 8 calls. A 4-call GM-ODE 2 (32 sub-steps; larger k and bias) on w1173 is untested before the real evaluations. | Replace `gm_ode` on the pixel model with `gm_ode2 4` on w1173 (same wall). |
| M2 | MEDIUM | `docs/gmflow_hn.md:59, 95, 104`; hn report `:17, :57, :167` | After the merge, they tell the reader to set `EVALS_WT`, which no longer exists. A gm_ode_cal job written from them runs on `gmflow_run2` and fails with "no spread-table code". | Rewrite with `--wt gmflow_hn_run` in the merge commit. |
| M3 | MEDIUM | `jobs/eval_gmflow.sh:30` (`EVAL_SPREAD`) | `wq submit eval_gmflow.sh 8 <step> gm_ode 8` on a pixel run cannot pass `EVAL_SPREAD` (no submit environment). It defaults to image, and the guard refuses it. That is safe but wastes a queue slot. | Go through `evals_gmflow.sh … --wt <wt> --spread pixel`; say so in the eval_gmflow.sh header. |
| M4 | MEDIUM (pre-existing) | `jobs/eval_gmflow.sh:24` | Without `GMFLOW_WT`, a direct submission evaluates the live dev worktree `gpic-wt/gmflow` at run time. | Default to refusing, or to the frozen worktree. |
| L1 | LOW | `test_gmflow.py` T11 | A spatially permuted offset (`transpose(-1, -2)`) passes T9 and T11–T14 (my mutant run). The code is right today. | Add the alignment check (offset bias r·16 + c vs the means' unpatchify). |
| L2 | LOW | `docs/gmflow.md` 7.3 | MS and C0/16 of the pixel runs (15552 / 16608 and 31104 / 33216) are not written; the evaluations must name them. | Add them. |
| L3 | LOW | `jobs/smoke_gmflow2.sh:83` | Any non-zero exit of the guard call (a timeout or a missing checkpoint) is logged as "guard refused". | Require the assert text in `eval_guard.log`. |
| L4 | LOW | `jobs/smoke_gmflow2.sh:44-50` | Recompiles after warm-up are counted, not failed. | Fail on > 0, or read them by eye. |
| L5 | LOW | `jobs/train_gmflow.sh:27`, `evals_gmflow.sh:25` | `--wt` takes any string; `--wt ../x` would leave `gpic-wt/`. | Check `[[ $WTN =~ ^[A-Za-z0-9_-]+$ ]]`. |
| L6 | LOW | `jobs/smoke_gmflow2.sh` timeouts | 40 min of timeouts and a 46.5 min worst case leave 3.5 min of slack under a 50 min wall. | Accept, or move to `--time=00:55:00`. |

Conditions for the real runs (all from the findings above, none needs a code change before the smoke):
1. A new frozen worktree at 251b3fa or later (for example `gpic-wt/gmflow_run3`), clean, selected with `--wt` in every training and evaluation call of group D. New tags, so no run root collides with `gmflow_k8_d640` / `gmflow_k1_d640` (w1173 / w1174: `commit.txt` = df37cea, no `spread.txt`).
2. The pixel runs: `train_gmflow.sh 8 d640 640 9 2 62208 <tag> w1147 --wt <wt> --spread pixel …` and `train_gmflow.sh 1 d640 640 9 2 66432 <tag> w1147 --wt <wt> --spread pixel …`; the evaluations at 31104 / 62208 and 33216 / 66432.
3. The order-2 evaluations of w1173 / w1174: `evals_gmflow.sh 8 d640 640 9 2 <new root> --link …/gmflow_k8_d640/exp_ladder_d640 --wt <wt> 62336:gm_ode2:8 …` (and 4 calls), never into w1173's own root (its `commit.txt` holds df37cea; `evals_gmflow.sh` would keep it and record nothing about the new code; the eval writes `ckpt.txt` with the new code sha, so the provenance is in each eval directory, but a new root is cleaner). Skip ≥ 32-call order-2 evaluations (H1).
4. The smoke must pass first: the train / resume checks, the three pixel predicts, the guard, and the three w1173 predicts.
5. Before merging `gmflow-hn` into `gmflow`: H3 (b) and M2.

VERDICT: GO
