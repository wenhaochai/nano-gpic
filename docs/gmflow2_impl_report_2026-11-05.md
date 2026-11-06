# GMFlow group D: per-pixel spread and second-order samplers, implementation report (2026-11-05/06)

Branch `gmflow`, worktree `gpic-wt/gmflow`, base `c16d71d`. Code in 4 commits: `9f4ccc8` (per-pixel spread), `30680d8` (second order), `501b024` (tests), `8604994` (job scripts). Docs in the commit that adds this file. Nothing was pushed. No job was submitted or cancelled. Other worktrees (`gmflow_run`, `gmflow_run2`) were not touched.

## Result

- Every CPU test passes: T1–T10 (unchanged, T9 extended) and the new T11–T14. The full run (vis1, about 5.5 min) is in `logs/gmflow2_tests.txt`, which is gitignored; the key lines are quoted below.
- All 10 mutants are caught: the 5 new ones (offset not detached; spread varies over channels; avg_var over the wrong axis; k without 0.5; no first-order fallback) and the 5 old ones.
- `gm_spread="image"` and `order=1` are bit-identical to `c16d71d`. For the model this covers parameters, RNG stream, outputs and gradients, at K = 8 and K = 1, in fp32 and under bf16 autocast. For the samplers it covers GM-SDE, GM-SDE final_mean and GM-ODE.
- FLOPs: the offset head adds 83.89 MFLOP per image. For gm_spread pixel, K = 8 gives r = 0.07252 and STEPS = 62208; K = 1 gives r = 0.00487 and STEPS = 66432. Both STEPS equal the owner's estimate.
- The older test files still pass: energy, implicit, plan A (code), AR.
- `torch.compile` (CPU, small model) with gm_spread pixel: 2 graphs over 6 trainer steps at 3 text lengths, the same as before, and no graph break.
- `jobs/smoke_gmflow2.sh` is written, syntax-checked and dry-run through stub sub-scripts. It is not submitted, so no code has run on a GPU yet.

## Files changed (lines at `8604994`)

| File | What |
|---|---|
| `baselines/PixelGen/src/models/transformer/JiT_T2I.py:332,342-350` | option `gm_spread` ("image" default, "pixel"); asserted to need `gm_k > 0` |
| `JiT_T2I.py:400-404` | `gm_spread_head = Linear(hidden, p^2)`, built after `initialize_weights` inside `torch.random.fork_rng`, so every other parameter and every later RNG draw equal the image model's; weight and bias zero |
| `JiT_T2I.py:562-567` | in `gm_mode="full"` with spread pixel: `h = norm_final(x)`, `x = linear(h)`, `spread = unpatchify(gm_spread_head(h.detach()))` → (B, 1, 1, H, W) in fp32. Other modes take the old line unchanged |
| `JiT_T2I.py:575-576` | `logstd = s-MLP(gm_cond)` (B, 1, 1, 1, 1), plus `spread` when it exists |
| `src/diffusion/flow_matching/gmflow.py:1-8,33-35,73-75` | docstrings: every function takes logstd (B, 1, 1, 1, 1) or (B, 1, 1, H, W); `gm_convert` does not reduce over pixels |
| `gmflow.py:94` | `GM2_CA = 0.005` |
| `gmflow.py:97-105` | `gm_iso_var`: the isotropic surrogate (GMFlow `gm_to_iso_gaussian`, then the mean over H, W as the no-guidance branch of `gm_2nd_order`) |
| `gmflow.py:108-149` | `gm_second_order`: the second-order reweighting (below) |
| `src/diffusion/flow_matching/sampling_gmflow.py:5-7,28-32` | option `order` (1 or 2) on both GM samplers |
| `sampling_gmflow.py:46-56` | `_second_order`: the first call passes the mixture through and stores (raw mixture, x, sigma, h); later calls convert the stored raw mixture to the current x_t (Eq. 10) and call `gm_second_order` |
| `sampling_gmflow.py:71,76-77` | GM-SDE: order 2 before the draw; a point model (tensor output) skips it |
| `sampling_gmflow.py:107,113-114` | GM-ODE: order 2 before the sub-steps; the sub-steps convert the second-order mixture |
| `src/diffusion/flow_matching/training_gmflow.py:55-56,63-73` | diagnostics: s_u = each sample's mean over pixels; new `gm/logsu_pix_mean` and `gm/logsu_pix_std` (the mean and the std over pixels of log s_u, batch mean) when the spread is per pixel; the smoke print adds them |
| `tools/gmflow/test_gmflow.py:29,385-397` | `SPREAD_BASE = "c16d71d"`; T9 at gm_spread image and pixel, K = 8 and 1, asserts the hand count and the STEPS |
| `test_gmflow.py:432-455` | helpers: `load_at` (a module from `git show <commit>:<file>`), model inputs, a random loss on the full output |
| `test_gmflow.py:457-518` | T11 |
| `test_gmflow.py:520-552` | T12 |
| `test_gmflow.py:554-665` | per-pixel toy data, landing metrics, T13 |
| `test_gmflow.py:668-712` | `ref_iso`, `ref_convert_to_mean`, `ref_gm_2nd_order`: transcriptions of the official code (read, never imported) |
| `test_gmflow.py:715-877` | random mixtures, T14 |
| `tools/gmflow/mutate.py` | 5 new mutants |
| `tools/gmflow/check_compile2.py` | new: compile check of gm_spread pixel and of order 2 |
| `jobs/eval_gmflow.sh:12-18,30-32,38-39,43-46,62-68,71-72,78` | samplers `gm_sde2` and `gm_ode2` (ceil(128 / calls) sub-steps, at least 2); `EVAL_SPREAD` (image default); K = 0 refuses order 2 and the pixel spread; the guard asserts that `gm_spread_head.weight` is in the checkpoint if and only if the spread is pixel; `--model.denoiser.init_args.gm_spread` is passed only for pixel, so the image command line is the old one; the sample check also prints the mean within-image std |
| `jobs/evals_gmflow.sh:14-29,44` | optional `--wt <worktree>` then `--spread pixel` after `--link`; exports `EVAL_SPREAD` |
| `jobs/train_gmflow.sh:9-13,24-29,36-53,58,65-66` | optional `--wt <worktree>` then `--spread pixel` after `<data>`; the run records `spread.txt` and refuses another spread (a run started before this option that already has a checkpoint is image); the training gets `gm_spread pixel`; the evaluations get the same options |
| `jobs/smoke_gmflow2.sh` | new smoke script (not submitted) |
| `docs/gmflow.md` section 7 | the summary in Chinese |

## Feature 1: the per-pixel spread

`log s_u(pixel) = s-MLP(detached t-embedding + mean caption embedding) + offset(pixel)`. Here `offset = unpatchify(Linear(hidden, 256)(norm_final(x).detach()))`.

- Input: the brief asks for "the detached features that feed the final linear (after the final norm and the adaLN modulation)". Our `FinalLayer` (`JiT_T2I.py:226-238`) has no adaLN modulation, only `norm_final` (LayerNorm without affine) and then `linear`. So the input is `norm_final(x)`, which is exactly what the final linear reads. I take it once and feed it to both linears, so the forward graph of the means and logits is the same as before.
- Zero init: weight and bias are 0, so the offset is 0 and the pixel model equals the image model (T11). The head is built after `initialize_weights()` and inside `fork_rng`. Otherwise its default `kaiming_uniform_` init would use RNG draws and shift the init of every later module, or of the next `torch.rand`. With `fork_rng` the pixel model's other parameters equal the image model's bit for bit (T11 checks this).
- Sharing: the offset has shape (B, 1, 1, H, W). It broadcasts over the K components (dim 1) and the C channels (dim 2). That is isotropic per pixel, as the paper's per-pixel GM factorization.
- Mean mode (`gm_mode="mean"`, used by Euler on the mean) does not compute the offset: it is not needed there, and that mode's FLOPs and graph stay unchanged.
- Optimizer: the new parameters are in `denoiser.parameters()`, so the existing AdamW group, EMA and gradient clip take them with no change (`lightning_model_gpic.py:149-158`).

### Every formula that takes s_x, with a per-pixel s_x

| Formula | Code | What it does with s_x (B, 1, 1, H, W) |
|---|---|---|
| `logstd_x` | `gmflow.py:33` | log s_u + log sigma; broadcasts |
| transition loss | `transition_nll`, `gm_nll` | the transition std `sqrt(c2^2 s_x^2 + c3)` per pixel; `gm_nll`'s inverse std and `- logstd` broadcast to (B, K, 1, H, W) and sum over C. The per-pixel NLL is then averaged over pixels (`training_gmflow.py`), as before: that is an average of per-pixel losses, not an average of variances. T13 checks it against a grid integral per pixel (2e-16). |
| GM-SDE draw | `gm_draw` | `mu + s_x xi`, s_x per pixel (T13: each pixel's mean and variance within 0.2%) |
| GM-SDE bridge step | `bridge_coefs` | does not involve s_x |
| GM-ODE sub-step | `gm_convert` | per pixel: `norm = g_var + s_x^2`, means, weights and the new variance (B, 1, 1, H, W); no reduction over pixels (T13: grid Bayes per pixel, 1e-15) |
| Euler on the mean | `gm_mean` | does not involve s_x |
| second order: surrogate | `gm_iso_var` | per pixel `sum_k A_k |mu_k - mean|^2 / C + s_x^2`, then **the mean over H, W** (below) |
| second order: product | `gm_second_order` | per pixel, s_x^2 in `denom = v' + s_x^2 beta` (T14: grid Bayes per pixel, 1e-15) |
| diagnostics | `training_gmflow.py:63-73` | s_u = per-sample mean over pixels; plus the mean and the std over pixels of log s_u |

**The image averages in the second order.** The official code (no-guidance branch) fits one isotropic Gaussian to the whole image. Its mean is the per-pixel mixture mean. Its variance `var` is the per-pixel variance (`gm_to_iso_gaussian_jit`: `sum_k A_k diff^2`, mean over C, plus `gm_vars`), averaged over H and W (`base_gaussian['var'].mean(dim=(-1, -2))`). `avg_var` is then the mean of `var` over (C, H, W), which equals `var`. With a per-pixel s_x I keep these averages, so the surrogate is still `N(mean, var I)` in D = C H W dimensions, as the paper's eq:gm2 (`s_+^2 I`). This means:
1. A per-pixel s_x enters the bias cap (`bias_power <= avg_var`), the error power (`avg_var * c_a`) and the ratio's variance only through its image average.
2. In the product of each component with the ratio Gaussian, s_x enters pixel by pixel. A pixel with a small s_x keeps its components (its means move by `s_x^2 (bias + beta mean) / (v' + s_x^2 beta)`, which is small). A pixel with a large s_x moves further toward `mean + bias`.
3. An image whose variance differs much across pixels gets a correction capped by the image average, which the high-variance pixels dominate.
4. I did not make the surrogate per pixel. That would be a new method, not the paper's. The mutant "avg_var over the wrong axis" (mean over the batch and W) is caught by T14, because T14 uses per-sample, per-pixel spreads that differ across samples.

## Feature 2: second order (GM-SDE 2, GM-ODE 2), no guidance

Per call `i` at sigma with step `h = sigma - sigma_next` (the last call: `h = sigma`, as the code):
1. Call 0: first order. Store (raw mixture, x, sigma, h) (official `gmflow.py:312-313`, `prev_gm is None`).
2. Call i > 0: `mean_prev = gm_convert_mean(prev raw mixture, prev sigma → sigma, prev x, x)` (Eq. 10; official `denoising_gm_convert_to_mean`). The stored mixture is the previous call's raw network output, not the corrected one: the code sets `self.prev_gm = gm_output` before it changes `gm_output`.
3. `mean, var = gm_iso_var(gm)`; `avg_var = mean of var`.
4. `k = 0.5 h / prev_h`; `err_power = avg_var * c_a` (c_a = 0.005; the guidance term `w^2 c_b` is 0); `mean_diff = (mean - mean_prev) * sqrt(max(0, 1 - err_power / prev_h^2)) * k`.
5. `bias_power = mean(mean_diff^2)` over (C, H, W); `bias = mean_diff * sqrt(min(1, avg_var / max(bias_power, 1e-6)))`; `v' = var * max(1e-6, 1 - bias_power / max(avg_var, 1e-6))`. As the code, `v'` uses `bias_power` before the cap.
6. `gm <- gm * N(x; mean + bias, v') / N(x; mean, var)`. Then the GM-SDE draw and bridge step, or the GM-ODE sub-steps (which convert this corrected mixture with `x_src` = the call's state, as the code's `denoising_gm_convert_to_mean(gm_output, x_t, x_t_base, ...)`).

### Where the paper and the code differ (I follow the code)

1. Scale of the correction. The paper rescales `Δμ` by `sqrt(max(0, 1 - (w^2 + c_a) s_c^2 / Δt^2))` with the current `Δt`. The code uses `prev_h_norm = prev_h / num_timesteps`, the previous call's step. On a uniform grid they are equal, except at the last call, where the code's `h = t` is unchanged and `prev_h` is the previous step.
2. The factor `k = 0.5 h / prev_h`. The paper writes `Δμ = (μ_+ - μ_-) / 2`. The code multiplies by `h / prev_h` as well, which is 1 on a uniform grid. We use the uniform grid, so this changes nothing in practice.
3. The cap of the bias power at the average variance. The code has it (`gmflow.py:284`); the paper does not. Without it, `s_+^2 - |Δμ|^2 / D` could be negative.
4. The quantities in variance. The paper writes `s_+^2 - |Δμ|^2 / D`. The code uses `var * (1 - bias_power / avg_var)` with `bias_power` before the cap and a floor of 1e-6. These are equal when var = avg_var (the no-guidance case) and the cap is inactive.

### One deviation from the code (stated, tested)

The code forms the ratio Gaussian `N(r_mean, r_var)` with `r_var = var v' / (var - v')` and clamps `var - v'` at 1e-6. It then multiplies each component by it with a clamp at 1e-6 on `r_var + s_x^2`. I write the same product without the division (`gmflow.py:116-120`): `beta = 1 - v'/var`, `mu_k' = (mu_k v' + s_x^2 (bias + beta mean)) / (v' + s_x^2 beta)`, `s_x'^2 = s_x^2 v' / (v' + s_x^2 beta)`, `log A_k' = log A_k - (beta |mu_k - mean|^2 - 2 bias . (mu_k - mean)) / (2 (v' + s_x^2 beta))`. Reasons:
1. When `bias = 0` (the clamp at 0 in step 4, for example the second call of a coarse grid where `err_power > prev_h^2`, or a mixture constant along the path), `var - v'` is 0. The code's clamp then gives `r_var = var^2 / 1e-6` and `r_mean = -1e6 mean / var * v'`-like terms, which pull every component mean toward 0 by about `1e-6 s_x^2 / var^2`. T14 measures this: 5e-6 in the "clamp at 0" regime. My form leaves the mixture exactly unchanged, which is what the paper's ratio (1 at Δμ = 0) says.
2. Everywhere else the two forms agree to 1e-15 (T14, "plain" and "bias cap" regimes, image and per-pixel spread).
3. An image with `var <= 1e-12` is returned unchanged. There the bias is capped to under 1e-6 RMS and `v'` could underflow. A point model (K = 1, s = 0) has var = 0. The GM-SDE path skips order 2 for a tensor output, and the GM-ODE path leaves it unchanged (T14).

### Not ported

The code's optional `gm2_correction_steps` (default 0 in `forward_test`), probabilistic guidance and the CFG branch. We never use CFG; the samplers still assert guidance 1.0.

## Tests (vis1, CPU, `tools/gmflow/run_vis.sh tools/gmflow/test_gmflow.py`)

```
T1 PASS T2 PASS T3 PASS T4 PASS T5 PASS T6 PASS T7 PASS T8 PASS T9 PASS T10 PASS T11 PASS T12 PASS T13 PASS T14 PASS ALL PASS
```

T9 (FlopCounterMode, d640, text length 8, batch 1, training forward against the point model):
```
K=8 spread image: point 31.0391 GFLOP, GMFlow 33.4731 (mean mode 33.4718); added 2434.01 MFLOP = r 0.07010 of 34.72; STEPS 62336 (x (1 + r) = 66706 <= 66816)
K=8 spread pixel: point 31.0391 GFLOP, GMFlow 33.5570 (mean mode 33.4718); added 2517.90 MFLOP = r 0.07252 of 34.72; STEPS 62208 (x (1 + r) = 66719 <= 66816)
K=1 spread image: point 31.0391 GFLOP, GMFlow 31.1243 (mean mode 31.1229); added 85.20 MFLOP = r 0.00245 of 34.72; STEPS 66560 (x (1 + r) = 66723 <= 66816)
K=1 spread pixel: point 31.0391 GFLOP, GMFlow 31.2081 (mean mode 31.1229); added 169.08 MFLOP = r 0.00487 of 34.72; STEPS 66432 (x (1 + r) = 66756 <= 66816)
```
The measured addition equals the hand count exactly: `2·256·640·(4·256·K − 768) + 2·(640·1024 + 1024) + 2·256·640·256` for pixel. The last term is 83.89 MFLOP (the brief's `2 · 256 · hidden · 256`). STEPS is the largest multiple of 128 with `STEPS (1 + r) <= 66816`. MS = STEPS / 4: 15552 (K = 8), 16608 (K = 1). C0/16 = STEPS / 2: 31104 and 33216.

T11 (bitwise and init):
```
K=8: image = c16d71d bitwise (outputs and gradients, fp32 and bf16); pixel at init = image; offset head trained, input detached; log s_u over pixels after a perturbation: std 0.3984
K=1: image = c16d71d bitwise (outputs and gradients, fp32 and bf16); pixel at init = image; offset head trained, input detached; log s_u over pixels after a perturbation: std 0.4111
```
- The c16d71d model is loaded with `git show` and `exec`. Both models start from the same seed: parameters, RNG stream after init, outputs (full and mean mode) and every parameter gradient of a random loss on the full output are `torch.equal`, in fp32 and under bf16 autocast, at K = 8 and K = 1.
- At init the pixel model's state dict is the image one plus the two zero tensors, and its outputs equal the image outputs (logstd broadcast) in fp32 and bf16.
- One trainer step: the offset head's weight and bias get a nonzero gradient. With all weights perturbed, a backward from `logstd` alone gives a nonzero gradient only to `gm_spread_head` and `gm_logstd`; every trunk parameter gets 0.
- The loss and the gradients are finite at t = sigmoid(±12).

T12 (four pixel groups with posterior std 0.10, 0.17, 0.25, 0.35; Adam on shared means and weights and one log s_u per group, given to the loss as a per-pixel logstd):
```
lambda 1.0, 2500 steps: s_x per group [0.1000, 0.1700, 0.2500, 0.3501] (largest relative error 0.000); means error 0.0002; weights [0.300, 0.700]
lambda 0.5, 6000 steps: s_x per group [0.0998, 0.1702, 0.2501, 0.3498] (largest relative error 0.002); means error 0.0005; weights [0.300, 0.700]
```
At λ = 0.5 the first version (2500 steps, 8192 pixels) ended at 0.1132 for the smallest std. I checked why (debug runs, not kept). At the true means, the loss differs by only about 1e-4 between s_x = 0.10 and 0.113, because `c3 = 0.083` is 20 times `c2^2 s_x^2` there: the loss is flat in s_x. Seeds 1 and 2 at 2500 steps recover it to 0.1% and 0.7%, and seed 0 does so at 6000 steps. So the test uses 6000 steps at λ = 0.5. This is a property of the λ = 0.5 transition loss, not a bug: small spreads are weakly identified at λ = 0.5.

T13 (per-pixel spread in the samplers):
```
GM-ODE sub-step per pixel: largest error against grid Bayes over 4 pixels with s_u [0.2, 0.4, 0.7, 1.1]: 1.02e-15
transition loss per pixel against the grid integral: largest error 1.67e-16
GM-SDE draw per pixel: largest error of mean / std and variance ratio 0.0023
GM-SDE 4 calls: f0 0.0011, dist0 0.0000, near0 1.0000, std0.02 0.0013, std0.05 0.0030
GM-SDE 50 calls: f0 0.0031, dist0 0.0000, near0 1.0000, std0.02 0.0018, std0.05 0.0001
GM-ODE 8 calls: f0 0.0022, dist0 0.0000, near0 1.0000, std0.02 0.1443, std0.05 0.0584
GM-ODE 50 calls: f0 0.0024, dist0 0.0000, near0 1.0000, std0.02 0.1243, std0.05 0.0499
```
- Data per pixel: `x0 ~ 0.3 N(P0, s0^2 I) + 0.7 N(P1, s0^2 I)`, with s0 = 0, 0.02, 0.05 by column. The network is the exact posterior, a K = 2 mixture whose logstd is (B, 1, 1, H, W).
- f0 = |share of P0 − 0.3|. dist0 / near0 = the largest distance to a data point, and the share within 0.05, at s0 = 0 pixels. std<s0> = |residual std / s0 − 1|.
- Every s0 = 0 pixel lands on a data point. GM-SDE gives each pixel group its own spread within 0.3%. GM-ODE within 15% (tolerance 20%). GM-ODE is a deterministic ODE with 128 Euler sub-steps, and the small spread shows its discretization error. The image-spread T5 already showed GM-ODE lands on the two points. This is the first test that measures GM-ODE's spread.

T14 (second order):
```
against the transcribed official code, largest mixture difference: image plain 7.3e-16; image clamp at 0 5.1e-06; image bias cap 4.4e-15; pixel plain 5.0e-16; pixel clamp at 0 5.8e-06; pixel bias cap 6.2e-15
reweighted mixture against grid Bayes per pixel: largest density error 1.18e-15 (V 0.4362, V' 0.4354)
exact posterior: |previous converted - current| / its std 1.3e-15; bias RMS / std 1.6e-16
order 1 = c16d71d bitwise (GM-SDE, GM-SDE final_mean, GM-ODE); first call of order 2 = first order; a point model is unchanged by order 2
two-point, GM-SDE 4 calls: order 1 f0 0.0009, dist0 0.0000, near0 1.0000 | order 2 f0 0.0009, dist0 0.0000, near0 1.0000
two-point, GM-SDE 8 calls: order 1 f0 0.0018, ... | order 2 f0 0.0018, ...
two-point, GM-ODE 4 calls: order 1 f0 0.0003, ... | order 2 f0 0.0003, ...
two-point, GM-ODE 8 calls: order 1 f0 0.0003, ... | order 2 f0 0.0003, ...
per-pixel GM, GM-SDE 4 / 8 calls and GM-ODE 4 / 8 calls: order 2 equal to order 1 (e.g. GM-ODE 8: std0.02 0.1392, std0.05 0.0544 both)
(information) one-Gaussian network, GM-SDE 4 calls: order 1: f0 0.034, dist0 2.257, near0 0.830, std0.02 5.977, std0.05 1.966 | order 2: f0 0.010, dist0 1.170, near0 0.578, std0.02 5.308, std0.05 1.379
(information) one-Gaussian network, GM-SDE 8 calls: order 1: f0 0.022, dist0 1.119, near0 0.999, std0.02 0.201, std0.05 0.061 | order 2: f0 0.001, dist0 1.115, near0 0.999, std0.02 0.247, std0.05 0.210
(information) one-Gaussian network, GM-ODE 4 calls: order 1: f0 0.008, dist0 1.498, near0 0.800, std0.02 6.506, std0.05 2.216 | order 2: f0 0.001, dist0 1.092, near0 0.741, std0.02 3.680, std0.05 0.939
(information) one-Gaussian network, GM-ODE 8 calls: order 1: f0 0.003, dist0 1.204, near0 0.994, std0.02 0.709, std0.05 0.342 | order 2: f0 0.001, dist0 1.344, near0 0.999, std0.02 0.064, std0.05 0.232
```
- The comparison with the code covers k, err_power, mean_diff, bias_power, bias, avg_var, v' and mean_prev at rtol 1e-9. It runs in float64 on random mixtures (B 3, K 4, 8 × 8, image and per-pixel spread) in three regimes. "Plain": no clamp active, asserted. "Clamp at 0": `err_power > prev_h^2`, the scale is exactly 0, and our mixture is unchanged. "Bias cap": the capped bias power equals avg_var. The reference (`ref_gm_2nd_order`) is a line-by-line transcription of `gm_2nd_order` (no-guidance branch), `gm_to_iso_gaussian_jit`, `denoising_gm_convert_to_mean_jit`, `iso_gaussian_mul_iso_gaussian_jit` (powers 1, −1) and `gm_mul_iso_gaussian_jit` (powers 1, 1), in the code's units (t = 1000 sigma, x-space logstds). It also covers the sampler path `_second_order`: the conversion of the stored raw mixture and the stored state.
- Zero correction: (a) `mean_prev = mean` gives a bias of exactly 0 and an unchanged mixture. (b) For the exact posterior, the previous call's mixture converted to the current x_t equals the current mixture to 1e-15, so the correction is 0. (c) A constant point mixture (K = 1, s = 0) is unchanged.
- Bitwise: the c16d71d `sampling_gmflow.py` and `gmflow.py` are loaded with `git show` and run with the same seed on the real small model (K = 3, perturbed weights). The trajectories and velocities of GM-SDE, GM-SDE final_mean and GM-ODE are `torch.equal` to the new code at `order=1`, both default and explicit. The first state of order 2 equals order 1, and the final state differs. The same holds on the gm_spread pixel model.
- Landing: with the exact posterior as the network, order 2 lands on the data at 4 and 8 calls, with errors no larger than order 1, on both the two-point data and the per-pixel GM data. They are equal, because the correction is 0 for the exact posterior (above). So this test shows that order 2 does not break the exact case; it cannot show a gain. The information lines use a network that is not the exact posterior: one Gaussian per pixel with the posterior's mean and variance. Their path is not self-consistent, so order 2 acts there. The share error drops in all four cases. The spread error drops for GM-ODE at 4 calls and at 8 calls (s0 = 0.02), and rises for GM-SDE at 8 calls. The direction is mixed, so I make no claim from it.

Mutation tests (`tools/gmflow/mutate.py`, each mutant on a temporary copy of `src`):
```
spread_input_not_detached: 'self.gm_spread_head(h.detach())' -> 'self.gm_spread_head(h)': failing tests ['T11']
spread_varies_over_channels: '... .unsqueeze(1)' -> '... .unsqueeze(1) * torch.arange(1, 4).view(1, 1, 3, 1, 1)': failing tests ['T11', 'T14']
avg_var_wrong_axis: 'var.mean((-1, -2), keepdim=True)' -> 'var.mean((0, -1), keepdim=True)': failing tests ['T14']
k_without_half: 'k = 0.5 * h / prev_h' -> 'k = h / prev_h': failing tests ['T14']
no_first_order_fallback: 'if prev is None: return gm, state' -> 'if prev is None: prev = state': failing tests ['T14']
softmax_over_channels: failing tests ['T7', 'T11']
c1_c2_swapped: failing tests ['T1', 'T3', 'T4', 'T5', 'T10', 'T12', 'T13', 'T14']
c3_dropped: failing tests ['T3', 'T5', 'T13', 'T14']
s_x_equals_s_u: failing tests ['T1', 'T3', 'T4', 'T6', 'T12', 'T13', 'T14']
no_bridge_noise: failing tests ['T5', 'T13', 'T14']
```
The brief names the spread mutant "broadcast over K instead of over the channels". With logstd (B, 1, 1, H, W) that mutant is a spread that varies over the channels and is shared over K, which is what I planted. "No first-order fallback" makes the first call use its own state as the previous one. It is caught by the first-call check, and by the zero-correction check on the real model.

Other checks:
- Older test files (vis2): `tools/energy/test_energy.py`, `tools/dist_step/test_implicit.py`, `tools/dist_step/test_code.py` print "all checks passed"; `tools/ar/test_ar.py` exits 0.
- `tools/gmflow/check_compile2.py` (vis1): the small pixel model with `net.compile()`, 6 trainer steps at text lengths 16, 24, 40, 24, 16, 40, diagnostics on and off: 2 graphs (the second when the text length becomes dynamic), then none, and no graph break. `logsu_pix_std` grows from 0 to 0.025, so the offset head trains. GM-ODE 2 (8 × 16) and GM-SDE 2 run on the compiled model with finite output.
- `main.py fit --print_config` with `gmflow_train.yaml` and `--model.denoiser.init_args.gm_spread pixel` resolves `gm_spread: pixel`. `main.py predict --print_config` with the ODE and SDE overlays and `--model.diffusion_sampler.init_args.order 2` resolves `order: 2` (and substeps, final_mean).
- Checkpoint guard (the heredoc extracted from `eval_gmflow.sh` and run as is): w1173's `ms-step=062336.ckpt` passes as image and is refused as pixel. A synthetic state dict with `gm_spread_head.weight` is refused as image and passes as pixel.
- Script dry run: `train_gmflow.sh` and `evals_gmflow.sh` with stub sub-scripts and a fake `git`, on scratch. The old calls pass the old arguments (no `gm_spread`, `EVAL_SPREAD=image`, worktree `gmflow_run2`). `--wt gmflow_new --spread pixel` passes `--model.denoiser.init_args.gm_spread pixel` to the training and `EVAL_SPREAD=pixel` plus the new worktree to the evaluations. A pixel run restarted as image is refused. `--spread pixel` on a worktree without the code is refused. A run of before this option (a checkpoint, no `spread.txt`) is recorded as image. `--spread bogus` is refused.

## Changes from the brief, and why

1. **`--wt <worktree>` in `train_gmflow.sh` and `evals_gmflow.sh`** (not asked). Both scripts hard-code the frozen worktree `gmflow_run2`, which holds c16d71d without the new code. A wq job runs in a clean environment, so a variable cannot select another worktree. Without `--wt`, `--spread pixel` could never run. The option comes right before `--spread`. The default is unchanged.
2. **The offset's input is `norm_final(x)`.** The brief says "after the final norm and the adaLN modulation". Our FinalLayer has no adaLN, so `norm_final(x)` is the input of the final linear.
3. **`fork_rng` around the head's construction** (not asked): it is needed so that every other parameter of the pixel model, and every RNG draw after init, equal the image model's. "At init the model equals the image model exactly" then holds for the whole model, not only for the output.
4. **The ratio Gaussian without the division by `var − v'`.** This differs from the code only where the code's clamp at 1e-6 acts. There the code moves the mixture by about 5e-6, and the exact ratio does not move it. Details above.
5. **The sample check prints the mean within-image std** (not asked), so a gray or noise-only sample shows in the smoke log.
6. **The smoke trains to 400 and resumes to 500 with `train_ladder.sh ... 1600`**: ms checkpoints every 400 steps, last.ckpt every 50. The smoke also evaluates `gm_ode` on w1173 (not asked) on the same prompts and noises as `gm_ode2`, so the effect of order 2 on a trained model reads as a paired difference. It also checks that the guard refuses the pixel checkpoint as image. Worst case under the timeouts: 46.5 min (limit 50).
7. **T12 at λ = 0.5 uses 6000 steps** (reason above).
8. **GM-ODE's spread tolerance in T13 and T14 is 20%** (measured 14%). GM-SDE's is 5% (measured under 0.3%).

## What I could not test

1. Nothing ran on a GPU: compile with inductor on CUDA, DDP, bf16 on H100, the resume of a pixel run, and predictions through `main.py`. The smoke test covers these.
2. The official code was not run (the rule forbids it). The reference in T14 is my transcription. A shared misreading would pass. The grid-Bayes check of the product (T14) and the paper's formulas are independent of it, but they do not cover k, err_power or the cap.
3. On the exact posterior the second-order correction is 0, so the CPU tests cannot show that order 2 helps a trained model. Only w1173's paired difference (smoke) and the FD (group D's evaluations) can.
4. The per-pixel spread's effect on FD is unknown until group D trains.
5. The step time of the pixel head: FLOPs add 0.25% over the image head. The extra fp32 (B, 1, 1, 256, 256) tensor is small next to the (B, 8, 3, 256, 256) loss tensors. The smoke measures it.

## Open risks

1. The offset head sees detached features, so the trunk never learns features for the spread. The offset can use only features that the means already use.
2. The second-order cap uses the image-average variance, so pixels with a large spread set the size of the correction for the whole image.
3. log s_u per pixel has no floor, as the per-image log s_u. GMFlow relies on the transition loss's c3 and on the gradient clip. T11 shows finite loss and gradients at t = sigmoid(±12).
4. Group D's real jobs need a new frozen worktree at `8604994` or later, selected with `--wt`. Its runs must use new tags, so `commit.txt` and `spread.txt` do not clash with group B–F runs.
