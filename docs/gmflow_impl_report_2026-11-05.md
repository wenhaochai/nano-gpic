# GMFlow head: implementation report (2026-11-05)

Branch `gmflow`, worktree `gpic-wt/gmflow`. Base `9711ff9`. Code at `ef1346e`; docs at the commit that adds this file. Nothing was pushed, and no job was submitted.

## Result

Every CPU test passes (T1–T10), and all 5 planted bugs are caught. The existing tests (energy head, implicit, plan A, AR) still pass. The `gm_k = 0` path is bit-identical to `9711ff9`, including under bf16 autocast.
The tests caught one real bug, which I fixed: the Gumbel noise in `gm_draw` was NaN (wrong sign), so every pixel drew component 0.
FLOPs: for K = 8, r = 0.07010 and STEPS = 62336. For K = 1, r = 0.00245 and STEPS = 66560.
The four job scripts are written and syntax-checked. They have not been submitted, so nothing has run on a GPU.

## Files changed (lines at `ef1346e`)

| File | What |
|---|---|
| `baselines/PixelGen/src/models/transformer/JiT_T2I.py:331,339-344` | option `gm_k`; asserts it is not combined with `head_width`, `xi_dim`, `code_dim` |
| `JiT_T2I.py:379-382` | head: `FinalLayer(hidden, K·4·p²)` and `gm_logstd` = SiLU → Linear(hidden, 1024) → SiLU → Linear(1024, 1) |
| `JiT_T2I.py:443-448` | init: the final Linear keeps `_basic_init`'s Xavier weight with a zero bias; the last s-MLP layer is zeroed (s_u = 1); `gm_k = 0` still zeroes the weight as before |
| `JiT_T2I.py:504-509` | `gm_unpatchify`: fold, then split into means (B, K, 3, H, W) and logits (B, K, 1, H, W) |
| `JiT_T2I.py:511-516` | `forward(..., gm_mode="mean")`: "mean" returns Σ A_k μ_k (B, 3, H, W); "full" returns the dict |
| `JiT_T2I.py:526-531` | s-MLP input: t-embedding + the y_embedder output averaged over real tokens (all tokens when y_mask is None), `.detach()`. `ypos` is cast to the input's dtype, as before |
| `JiT_T2I.py:549-555` | log_softmax over K in fp32; the mean or full output |
| `src/diffusion/flow_matching/gmflow.py` (new) | `bridge_coefs` :12, `forward_transition` :24, `logstd_x` :31, `gm_nll` :36, `transition_nll` :45, `gm_mean` :55, `gm_draw` :60, `gm_convert` :71 (Eq. 10, port of gmflow.py:41-70), `gm_convert_mean` :86 |
| `src/diffusion/flow_matching/training_gmflow.py` (new) | `GMFlowTrainer(JiTTrainer)`: t from the baseline sampler :28-33; σ_high, σ_low, x_low, x_high in fp32 with autocast off :34-40; net(x_high, t) :41-43; loss :44-48; diagnostics :49-50, :58-73; `GPIC_GM_PRINT` smoke print :51-55 |
| `src/diffusion/flow_matching/sampling_gmflow.py` (new) | `as_mixture` :14; `_GMSampler` (CFG assert, per-rank seed) :24-39; `GMSDESampler` :42-69; `GMODESampler` :72-101 |
| `src/lightning_model_gpic.py:135-138` | pre-clip grad norm logged on log steps; printed on `GPIC_GM_PRINT` steps |
| `src/lightning_model_gpic.py:220-222,234` | `_log_now` computed once; it sets `diffusion_trainer.diag_now` |
| `src/lightning_model_gpic.py:290-291` | the masked-text predict wrapper forwards `**kw` (gm_mode) |
| `configs_h100/gmflow_train.yaml` | copy of `energy_train.yaml`: `gm_k: 8` instead of `head_width`; `GMFlowTrainer` (lam 0.5, weight_scale 2.0) instead of `ImplicitJiTTrainer`; same t sampler and `null_condition_p: 0.0` |
| `configs_h100/gmflow_{sde,ode,euler}_sample.yaml` | sampler overlays on `lite_ladder_sample.yaml` |
| `jobs/smoke_gmflow.sh`, `run_gmflow.sh`, `eval_gmflow.sh`, `eval_point_sampler.sh` | J1–J4 |
| `tools/gmflow/test_gmflow.py`, `mutate.py`, `run_vis.sh` | tests, mutation driver, vis-node runner |

## Design as built, in our convention

- t = 1 is data. σ := 1 − t and α := t. s_x = σ s_u.
- Loss: σ_high = clamp(1 − t, 1e-4, 1), σ_low = (1 − λ) σ_high, x_low = (1 − σ_low) x + σ_low ε₀, x_high = forward_transition(x_low). The network gets t_net = 1 − σ_high, which equals t except when 1 − t < 1e-4.
- Per pixel: −logsumexp_k [log A_k + Σ_c (−½ ((x_low − c1 x_high − c2 μ_k)/σ_T)² − log σ_T)], with σ_T² = c2² s_x² + c3. Then the mean over pixels, × 2, and the mean over the batch. Inverse std clamped at 1e4, σ ≥ 1e-4. No t_eps.
- `logaddexp` combines the two variance terms. When c3 = 0 (λ = 1), `log(0) = -inf`, and logaddexp returns the first term exactly. So λ = 1 reduces to the plain NLL (T3).
- GM-SDE: x0̂ ~ mixture per pixel (Gumbel-max over K, then + s_x ξ). Then x' = c1 x + c2 x0̂ + √c3 ξ'. On the last step, c1 = 0, c2 = 1 and c3 = 0, so the step returns x0̂, or the mixture mean with `final_mean`.
- GM-ODE: on each call, sub-step 0 uses the mixture mean. Sub-step j > 0 uses `gm_convert_mean(gm, σ_src, σ_j, x_src, x_j)`, with σ_src the call's σ and x_src its state. Each sub-step is an Euler step x += (σ' − σ)(x − x0)/σ on a uniform σ grid of N·n points. This follows gmflow.py:527-539: `x_t_base`, `t_base` stay fixed for the call.
- Point model in GM-SDE: the tensor output is x0̂ (K = 1, s = 0), so this is DDPM-small. JiT_T2I with gm_k = 0 accepts `gm_mode` and ignores it, so the call signature is the same.
- Per-rank seed in the GM samplers: `torch.manual_seed(1000 + LOCAL_RANK)` once, as in `EulerSamplerJiT`'s eta path. Without it, all 8 ranks would draw the same per-pixel noise for batch i.

## Test output (vis2, CPU, `tools/gmflow/run_vis.sh tools/gmflow/test_gmflow.py`; warnings removed)

```
T1: K = 1, s_u fixed (s_x = c (1 - t)), lambda = 1: the gradient on mu_x is the baseline v-loss gradient times one constant, C / c^2, where 1 - t >= t_eps.
  least-squares ratio of the gradients on mu outside the clamp: 6.122448 (expected C / c^2 = 6.122449); largest deviation 1.5e-06 of the largest gradient; 64 samples, sigma in [0.252, 0.970]
T1 PASS
T2: The NLL written over x and written over v (u = (x_t - x) / sigma) differ by C log sigma per pixel, a constant in theta, so their gradients are equal.
  NLL_x - NLL_u = C log sigma to 6.66e-15; gradients equal to 8.88e-15
T2 PASS
T3: c1, c2, c3 in our convention: a Monte Carlo of the two-stage forward process with a fixed x0 gives the bridge's mean and variance; lambda = 1 gives c1 = 0, c2 = 1,
  sigma 0.9 -> 0.450: MC slope 0.0456 vs c1 0.0455; intercept 0.2020 vs c2 x0 0.2018; residual var 0.20070 vs c3 0.20083
  sigma 0.5 -> 0.250: MC slope 0.1667 vs c1 0.1667; intercept 0.2469 vs c2 x0 0.2467; residual var 0.05550 vs c3 0.05556
  sigma 0.3 -> 0.225: MC slope 0.5082 vs c1 0.5081; intercept 0.1550 vs c2 x0 0.1552; residual var 0.02741 vs c3 0.02739
  sigma 0.97 -> 0.194: MC slope 0.0014 vs c1 0.0015; intercept 0.2984 vs c2 x0 0.2982; residual var 0.03767 vs c3 0.03763
  lambda = 1: c = (0, 1, 0); trainer loss 6.495058 = 2 x plain NLL 6.495058
T3 PASS
T4: The transition loss (lambda = 0.5) is minimized at the truth: for a fixed x_high with a known two-mode posterior, Adam on (mu_k, a_k, log s_u) from a random init
  means [[-0.4950000047683716, 0.19599999487400055, 0.7960000038146973], [0.5960000157356262, -0.3970000147819519, 0.10300000011920929]] vs truth [[-0.49399998784065247, 0.19699999690055847, 0.7960000038146973], [0.5950000286102295, -0.39800000190734863, 0.10300000011920929]]; weights [0.289000004529953, 0.7110000252723694] vs [0.289000004529953, 0.7110000252723694]; s_x 0.1503 vs 0.1493
T4 PASS
T5: GM-SDE with the exact posterior reproduces the 2-point data at 4 and 50 calls (every pixel on a data point, P[0] at 0.3), and every intermediate state has the exact
  GM-SDE 4 calls: on a data point 1.0000 (max distance 0.0e+00), share of P[0] 0.3024, largest marginal error 0.0121
  GM-SDE 50 calls: on a data point 1.0000 (max distance 0.0e+00), share of P[0] 0.3011, largest marginal error 0.0135
  GM-ODE 50 calls x 3 sub-steps: on a data point 1.0000, share of P[0] 0.2989
  GM-ODE 8 calls x 16 sub-steps: on a data point 1.0000, share of P[0] 0.2988
  control, Euler on the mean, 4 calls: on a data point 0.9075, share of P[0] 0.2626
  point model: GM-SDE = DDPM-small, step by step
T5 PASS
T6: The GM-ODE sub-step: the closed-form re-weighted mixture (gm_convert) matches numerical Bayes on a grid, 1-D, K = 2.
  grid vs closed form: largest density error 6.80e-16 of the peak; mean 0.045510 vs 0.045510
T6 PASS
T7: Shapes and modes: a planted pattern comes back at its pixel, channel and component after unpatchify; the weights sum to 1 over K; the mean mode is sum A mu;
  planted pattern found at its pixel, channel and component; weights sum to 1; mean mode = sum A mu; gm_k = 0 bit-identical to 9711ff9
T7 PASS
T8: Init: the K means differ; s_u = 1; the loss and the gradients are finite for t near 0 and near 1; the s-MLP's input is detached; the diagnostics appear only when asked.
  t = sigmoid(-12.0): loss 0.1960, n_eff 7.946, max weight 0.126
  t = sigmoid(12.0): loss 66417024.0000, n_eff 7.946, max weight 0.126
  init: spread of the K means 0.120, s_u = 1; s-MLP input detached; no diagnostics unless asked
T8 PASS
T9: FLOPs: FlopCounterMode on the d640 model (text length 8), the GMFlow head (training mode, gm_mode full) against the point model; the added FLOPs equal the hand count.
  K=8: point 31.0391 GFLOP, GMFlow 33.4731 (mean mode 33.4718); added 2434.01 MFLOP = r 0.07010 of 34.72; STEPS 62336 (x (1 + r) = 66706 <= 66816)
  K=1: point 31.0391 GFLOP, GMFlow 31.1243 (mean mode 31.1229); added 85.20 MFLOP = r 0.00245 of 34.72; STEPS 66560 (x (1 + r) = 66723 <= 66816)
T9 PASS
T10: Extra: GM-ODE with one sub-step is the exact DDIM step on the mixture mean (ImplicitSampler on the mean mode); GM-SDE final_mean ends on the mixture mean;
  GM-ODE n = 1 = exact DDIM on the mean; final_mean ends on the mean; CFG asserted
T10 PASS
ALL PASS
```

Mutation tests (`tools/gmflow/mutate.py`). Each bug goes into a temporary copy of `src`, and the full suite runs on that copy. The worktree is never edited, so no revert is needed:

```
softmax_over_channels: models/transformer/JiT_T2I.py: 'logweights = logits.float().log_softmax(dim=1)' -> 'logweights = logits.float().log_softmax(dim=2)': failing tests ['T7']
c1_c2_swapped: diffusion/flow_matching/gmflow.py: 'return c1, c2, c3' -> 'return c2, c1, c3': failing tests ['T1', 'T3', 'T4', 'T5', 'T10']
c3_dropped: diffusion/flow_matching/gmflow.py: 'c3 = beta_over_sigma_sq * sigma_to ** 2' -> 'c3 = 0 * beta_over_sigma_sq * sigma_to ** 2': failing tests ['T3', 'T5']
s_x_equals_s_u: diffusion/flow_matching/gmflow.py: 'return gm["logstd"] + torch.log(sigma.clamp(min=EPS))' -> 'return gm["logstd"] + 0 * sigma': failing tests ['T1', 'T3', 'T4', 'T6']
no_bridge_noise: diffusion/flow_matching/sampling_gmflow.py: 'c3.clamp(min=0).sqrt() * torch.randn_like(x)' -> '0 * torch.randn_like(x)': failing tests ['T5']
every mutant killed
```

Existing tests, same vis node:

```
tools/energy/test_energy.py      all checks passed; d640 forward GFLOP (CPU counter): point 31.039, energy m=2 31.714, head 0.9263, added 0.6746 = 1.94% of 34.72
tools/dist_step/test_implicit.py all checks passed
tools/dist_step/test_code.py     all checks passed
tools/ar/test_ar.py              all checks passed
```

Other checks:
- `main.py fit/predict --print_config` resolves all four overlays (`GMFlowTrainer` lam 0.5, `null_condition_p 0.0`, `gm_k`, `final_mean`, `substeps`). LR, EMA decay, clip and batch match `lite_ladder.yaml`.
- `torch.compile` of the denoiser on CPU (small model, as `net.compile()` in production), four trainer steps at text lengths 16, 24, 40, 24, with diagnostics on and off: 2 graphs (the second when the text length becomes dynamic), then no new graph, and no graph breaks. The mean mode adds one graph.
- `eval_gmflow.sh`'s checkpoint guard on w1147's `ms-step=033408.ckpt`: K = 0 passes, and K = 8 stops ("final layer has 768 outputs").

## FLOPs and STEPS

Method of docs/energy_head.md: `FlopCounterMode` on the d640 model (text length 8, batch 1), the training forward (`gm_mode="full"`) against the point model. r = added FLOPs / 34.72 GFLOP.

| K | point GFLOP | GMFlow GFLOP | added MFLOP | hand count | r | STEPS | STEPS·(1+r) | MS | C0/16 step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 31.0391 | 33.4731 | 2434.01 | equal | 0.07010 | 62336 | 66706 | 15584 | 31168 |
| 1 | 31.0391 | 31.1243 | 85.20 | equal | 0.00245 | 66560 | 66723 | 16640 | 33280 |

- Hand count: 2·256·640·(K·4·256 − 768) for the wider final Linear, plus 2·(640·1024 + 1024·1) for the s-MLP.
- The mean mode costs 1.3 MFLOP less, because it skips the s-MLP.
- The fp32 loss is elementwise work, and the counter does not count it. Like the energy head, the ladder scores matmul FLOPs only.
- Training counts forward × 3 on both arms, so r carries over.

## Changes from the brief, and why

1. **Lambda = 1 path.** The brief sets σ_low = (1 − λ) σ_high. I dropped GMFlow's `min(t_low, t_high − eps)`. In GMFlow's units (t ∈ [0, 1000], eps 1e-4) it acts only at σ_high < 2e-7, which lies below our σ ≥ 1e-4 clamp. It would also stop λ = 1 from giving σ_low = 0 exactly, and so from the exact reduction to the plain NLL.
2. **`t_eps: 0.05` stays in `gmflow_train.yaml`.** The loss does not use it. It is kept so that the trainer's arguments match energy_train.yaml.
3. **The point model calls with `gm_mode`.** JiT_T2I with gm_k = 0 accepts and ignores `gm_mode`, so one sampler code path serves both models. This adds no behaviour to the off path (T7: bit-identical).
4. **Extra test T10.** It checks: GM-ODE with n = 1 equals the exact DDIM step on the mixture mean; `final_mean`; the CFG assert; the 8 × 16 path on the real small model.
5. **T1 compares tensors, not elementwise ratios.** Where μ is close to x, a per-element ratio is fp32 noise. I compare the least-squares ratio and the largest deviation of the whole gradient: 6.122448 against C/c² = 6.122449, with deviation 1.5e-6 of the largest gradient. The ratio is C/c² because the NLL sums over 3 channels and doubles, while the baseline averages over channels.
6. **T5 checks the marginals at every step**, not only the final points. The largest error is 0.0135 of the std at 4 and at 50 calls.
7. **The real run evaluates more than "GM-SDE and the official Euler".** It runs GM-SDE, GM-ODE and Euler on the mean, at 50 and 8 calls, at both compute points (12 evals). This follows the owner's decisions 2 and 4 in docs/gmflow.md section 2 (commit 0022197): 50 and 8 calls are judged apart, and every sampler is evaluated. `--time` is 02:45 (energy: 02:00 with 4 evals).
8. **`eval_gmflow.sh` takes K as its first argument** (8, 1 or 0) and a fifth sampler `euler` for the point model. K selects the run root and the checkpoint guard. J4 uses K = 0.
9. **Smoke predictions use `EVAL_LIMIT_BATCHES=2`**: 2 batches × 32 × 8 ranks = 512 prompts, 8 calls, sample sanity check, no FD. With 8 calls, GM-ODE gets 16 sub-steps.
10. **The pre-clip gradient norm is logged** (`train/grad_norm_preclip`) on log steps. It was already computed in `configure_gradient_clipping`, so this adds no sync on the other steps. The smoke test prints it, because W&B is off there.

## What I could not test

- Nothing ran on a GPU: Lightning compile with DDP, CUDA graphs and kernels, resume, predict through `main.py`, the step time and memory, and the FD pipeline. The smoke test (J1) is the first check of all of these.
- Wall time. The (B, 8, 3, 256, 256) fp32 loss and its backward are memory-bound. A step may cost more than the 7% FLOPs. The smoke test reports s/step.
- Recompiles with real DDP: the CPU compile check had no DDP and no `CompileWarmup`. Only the denoiser is compiled, as in production. The diagnostics run in the uncompiled trainer, so `diag_now` does not change the denoiser's graph (in the CPU check, the second graph came from the dynamic text length, not from the diagnostics). The smoke test counts recompiles after `[warmup] done`.
- Behaviour at t → 1 in real training. T8 shows a finite loss and finite gradients at t = 1 − 6e-6 with the init s_u = 1 (loss about 7e7). Under the baseline t sampler, P(t > 0.95) ≈ 1.4e-6 and P(t > 0.99) ≈ 8e-12, so such batches are rare. Gradient clipping at 1.0 bounds the step.
- The FD effect of independent per-pixel sampling, and of RGB with C = 3 (digest section 12, notes 5 and 9). Only the real run can show it.
- I did not run the reference repo, by rule. The port was checked against its code by reading and by the tests above.

## Open risks

1. GM-SDE adds independent per-pixel noise. The paper's own ablation without spectral sampling went from FID 6.96 to 8.98. `gm_sde_final_mean` and GM-ODE are the checks.
2. With 3 RGB channels per pixel, pi-Flow's C = 1 toy showed over-saturation and smoothing.
3. The loss weighting changes from 1/(1 − t)² to about 1/MSE(t), capped by c3. The K = 1 run isolates this.
4. Seeds: as in w1147, all 8 ranks draw the same t and noise in training (`GPIC_RANK_SEED` unset; energy review F2). The comparison stays fair because w1147 does the same.
5. In predict, the draws depend on the rank and the batch position, not on the prompt (energy review F3). FD is unaffected, but the same prompt gives another image with another GPU count.
