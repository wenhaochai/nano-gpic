# Review: one mixture per image (gm_scope image), uncommitted change on gmflow @ 9642fed

Reviewer scope: `git diff` (JiT_T2I.py, gmflow.py, sampling_gmflow.py, training_gmflow.py, train/eval/evals_gmflow.sh), new `jobs/smoke_image_mix.sh`, T32 in `tools/gmflow/test_gmflow.py`, docs/gmflow.md section 15.
Planned run: `train_gmflow.sh 8 d640 640 9 2 62336 c08imagemix w1147 --wt gmflow_run6 --scope image <8 eval tasks>`, compared with w1173 (`runs/gpic-speed/gmflow_k8_d640`) and the point model.

What I ran (all CPU on vis2; scripts in `/scratch/gpfs/GROUP/USER/tmp/ladder_prof/review_image_mix_scripts/`):
- `test_gmflow.py T32`: PASS (output below, under "checks that passed").
- fp32 vs fp64 whole-image loss at 4x8x3x256x256, 5 values of sigma_high (inline script).
- `grad_scale.py`: the real d640 JiT_T2I at 256x256, one GMFlowTrainer step, gradient sizes of the final linear's mean rows and logit rows, pixel scope vs image scope.
- `collapse_toy.py`, `dead_mass.py`: a small JiT_T2I (64x64) trained with the real GMFlowTrainer, gm_scope image, K = 8, AdamW as lite_ladder (betas 0.9/0.95, eps 1e-8, clip 1.0), on clustered data (4 templates) and on unclustered smooth images.

## Blockers

None found. The code does what section 15 says. The run will not crash on these paths, and the comparison with w1173 is set up correctly (details under "checks that passed"). The main risk is in the method, not the code: see S1 and S2.

## Should-fix

### S1. Winner-take-all is likely, and the logged diagnostics can hide it. Add one cheap number: the weight on components that win nothing.
`training_gmflow.py:66-71` (win_stats), `training_gmflow.py:75-94` (diagnostics).

Why it is likely:
- The responsibilities are r_k ∝ A_k exp(sum over 196608 dims of log N). The log-likelihood gaps between components grow with D / variance. My fp64 measurement at 256x256 (means = 0.8 x + 0.2 noise): the smallest top-2 gap was 14 nats at sigma_high 0.9, 5 nats at 0.5, 125 at 0.1, 3.4e4 at 0.01 and 5.6e7 at 1e-4. So r is one-hot at every sigma except the noisiest. In fp32, r is exactly 0 once the gap is above about 104 nats.
- The gradient on mu_k is r_k times the component's own gradient (T32 checks this). A component that loses every image gets exactly zero gradient on its final-layer rows. This is hard-EM / k-means with no way back for a dead component except drift of the shared backbone. The winner gets better, so it wins more ("rich get richer").
- At the d640 init (one step, B = 4), the median gradient of the mean rows was exactly 0 (at most 4 winners of 8, so this evidence is weak).

The toy runs (CPU, real trainer, `collapse_toy.py`; win_max = largest share of the batch won by one component, alive = components that win at least 1 of 32 images):

| data, lr | step 0 | step 100 | step 200 | step 300 |
|---|---|---|---|---|
| smooth (no clusters), 1e-4 | win_max 0.22, alive 8, n_eff 8.00 | 0.88, 3, n_eff 7.88 | **1.00, 1**, n_eff 7.39 | 0.94, 3, n_eff 5.84 |
| smooth, 1e-3 | 0.22, 8 | 0.97, 2, n_eff 2.14 | 0.88, 3, n_eff 1.89 | 0.88, 3, n_eff 1.60 |
| 4 clusters, 1e-4 | 0.41, 8 | 0.34, 3, n_eff 7.79 | 0.50, 3, n_eff 7.53 | 0.53, 3, n_eff 6.98 |
| 4 clusters, 1e-3 | 0.41, 8 | 0.47, 4, n_eff 4.95 | 0.47, 3, n_eff 3.42 | 0.44, 3, n_eff 2.65 |

- On data with 4 clusters, only 3 components stay alive: 2 clusters merge into one component.
- **n_eff and max_w do not show the collapse.** At lr 1e-4, n_eff is 7.4-7.9 while 1-3 components do all the work.
- **The dead components keep their weight.** `dead_mass.py` (clusters, 300 steps, 256 fresh images): the components that win no image at any sigma hold 34-43% of the mean weight A (0.343 at sigma 0.9 and 0.432 at sigma 0.1; wins [0, 0, 124, 0, 62, 70, 0, 0], mean A of each dead component 0.067-0.091).

Why this matters for the comparison:
- gm_pick picks by A. If dead components keep weight, gm_draw sends a share of the samples toward untrained candidate images.
- euler_mean uses sum_k A_k mu_k, so it mixes the untrained means into every sample.
- The dead logits do get pushed down: their gradient is A_k - 0 > 0 every step. But that gradient is tiny (S2), so the decay is slow. The C0/16 checkpoint (31168 steps) is more at risk than C0/8.
- If the run fully collapses to 1 component, gm_draw becomes the plain Euler step on the x prediction (x' = x + (σ - σ')(μ - x)/σ). The run then measures "a point model with an unused s-MLP", not "one mixture per image". That is a valid result, but only if the logs show the collapse.

Fix (cheap, in win_stats, no extra sync beyond the one it already has):
```python
A = gm["logweights"].exp().flatten(1).mean(0)                      # (K,)
dead = torch.bincount(wins, minlength=K) == 0
out["gm/w_dead"] = (A * dead).sum()                                 # weight on components that win nothing in this batch
```
Also print it in the GPIC_GM_PRINT line, and add it to the smoke's regex. Read it at the first W&B points: alive ≤ 2 or w_dead > 0.2 after a few thousand steps means the run is collapsing. Note: win_stats sees only rank 0's 32 images (sync_dist False), so alive is a noisy lower bound out of 32. If the owner wants a mitigation, the usual ones are a temperature or a floor on r for the first N steps (soft EM, then hard), or a small share of the gradient for the losers (relaxed multiple-choice learning). Each is a new experiment, so each needs a popup.

### S2. The logits' gradient is about 2000x smaller than in the per-pixel model, close to Adam's eps 1e-8.
`JiT_T2I.py:579-581`, `gmflow.py:58-60`.

- The image's logit is the mean of 65536 per-pixel logits, and the loss is divided by n_pix = 65536. So each final-layer logit weight gets about (A_k - r_k) x 256 / n_pix² x mean(h). The factor 1/n_pix comes in twice.
- Measured at d640, 256x256, init, B = 4 (`grad_scale.py`; my printout scaled by 4/256, here unscaled):

| rows of the final linear | pixel scope | image scope |
|---|---|---|
| mean rows, median abs grad of W | 4.6e-5 | 0 (losers) |
| logit rows, median abs grad of W | 1.5e-5 | **8.1e-9** |
| logit rows, max abs grad of W | 2.0e-4 | 3.3e-8 |
| logit rows, median / max abs grad of the bias | 2.3e-5 / 1.9e-4 | 1.5e-8 / 4.5e-8 |

- AdamW's update is m / (sqrt(v) + 1e-8). With |g| ≈ 1e-8 the step is about half of lr. A batch of 256 averages (A_k - r_k) over images, so the real gradient is smaller than in this B = 4 measurement, and the damping is stronger.
- The backbone's share of the logit gradient is also 2000x below the means' share. In effect the weights A are learned at a much lower rate than the means. This is the reason the dead components' A decays so slowly in S1.

The run does not need to change for this. Record it in docs section 15 as a known property, so a stuck A is read correctly. Watching gm/w_dead (S1) shows its effect. If it proves to be a problem, one option is to compute the K logits from a pooled feature (a small linear on the mean token) instead of averaging the per-pixel logits. That is a model change and needs a popup.

## Minor

### M1. win_stats syncs the host through torch.bincount on CUDA.
`training_gmflow.py:70`. On CUDA, bincount computes max(input)+1 on the host, so it syncs. It runs only on diag_now steps (every 100) and on GPIC_GM_PRINT steps. Lightning already syncs on those steps to write the logs, so the extra cost is close to zero. The docstring at line 24 ("with no host sync") is now slightly wrong. Fix: `torch.zeros(K, device=d).scatter_add_(0, wins, torch.ones_like(wins, dtype=torch.float))` or `F.one_hot(wins, K).sum(0)`.

### M2. train_gmflow.sh does not check the eval tasks before the 8-GPU training.
`jobs/train_gmflow.sh:93-95`. With `--scope image`, a task with gm_ode or gm_sde runs all 62336 steps and is refused only afterwards, by eval_gmflow.sh (rc 2, so the wq job fails after the training). The same holds for a misordered option (for example `--scope image --lam 0.5`: `--lam` becomes a task and evals_gmflow.sh fails with "is not step:sampler:calls"). The planned command line is correct, so this does not affect the planned run. Fix: before training, loop over "$@" and refuse what eval_gmflow.sh would refuse (format step:sampler:calls; with scope image, sampler in gm_draw|euler_mean).

### M3. An older worktree's eval would load an image-scope checkpoint as per-pixel with no error.
`jobs/eval_gmflow.sh:94-97`. The scope has no weights, so only scope.txt protects it, and only the new eval_gmflow.sh reads scope.txt. `evals_gmflow.sh --wt gmflow_run5 ...` on the new root would give a per-pixel FD with no error. Nobody plans this. Keep `--wt gmflow_run6` for every evaluation of this root and say so in docs section 15.

### M4. The training loss of the two scopes is not the same quantity.
`gmflow.py:58-60`. The image-scope train/loss and gm/nll (−log Σ_k A_k Π N / n_pix) are not comparable with w1173's per-pixel loss, except at K = 1. Use FD only when comparing the two runs. Worth one sentence in section 15.

### M5. The smoke script is untracked.
`jobs/smoke_image_mix.sh`. It reads `$WT/jobs/train_ladder.sh` and `$WT/jobs/eval_gmflow.sh` from gmflow_run6, and needs `def gm_pick` in that worktree. Commit it with the change. Its check `git status --porcelain --untracked-files=no` ignores untracked files, so it does not catch this. Its 45-min limit is tight but enough: worst case is about 480 + 300 + 3x300 + 2x120 + 300 s ≈ 37 min of timeouts.

### M6. What T32 does not test (not wrong, just narrower than its docstring suggests).
- Part 4 (trainer): K = 1 equality is trivial (log_softmax over 1 component is 0 in both scopes). For K = 8 it checks only that the step runs, the keys exist and the gradient is non-zero.
- Part 3: transition_wins is checked against transition_nll with K = 1. Both use gm_component_ll, so the two are not independent. Part 2's fp64 brute force covers gm_component_ll, so the chain is covered.
- Nothing checks the 1/n_pix² size of the logit gradient (S2), the collapse (S1), or GPU / compile / DDP. The smoke covers the last three.
- image_scope() is a shape test: weights (…, 1, 1) over means that are not 1x1. I checked every mixture the code and tests build (as_mixture and point_mixture use (B, 1, 1, H, W); T6 has 1x1 means; T-lines 187 and 555 have W = n). None of them hits the new branch by accident. A per-pixel K > 1 mixture with weights broadcast as (B, K, 1, 1, 1) would now be scored as an image mixture. No caller builds one today.

## Checks that passed

1. **T32**: PASS on vis2. Output: model weights = log_softmax of the averaged per-pixel logits; loss = fp64 brute force (1e-12); K = 1 = per-pixel loss; gradient = r_k (mu_k − x)/(s² n_pix); winners = argmax r; gm_pick shares 0.201/0.295/0.504 for weights 0.2/0.3/0.5; the draw sampler on exact whole-image posteriors returns data images exactly at 1, 4 and 50 calls (largest marginal error 0.026); the posterior-mean control lands ≥ 1.32 from every image; script plumbing passes.
2. **Math of the whole-image loss** (`gmflow.py:51-61`): −logsumexp_k(Σ_pix ll_k + log A_k)/n_pix, returned as (B, 1, 1). The trainer's `nll.mean((1, 2))` is then the identity, and `weight_scale` and the batch mean apply as before. The gradient on log A_k is −r_k/n_pix; through log_softmax and the pixel mean it becomes (A_k − r_k)/n_pix per image logit. This is correct (see S2 for its size).
3. **Shapes and broadcasting**: logweights (B, K, 1, 1, 1) → squeeze(2) → (B, K, 1, 1); the branch test `lw.shape[-2:] == (1, 1) and ll.shape[-2:] != (1, 1)` matches image_scope. ll (B, K, H, W).sum((-2, -1)) + lw.flatten(1) gives (B, K). transition_nll's means c1 x_high + c2 mu broadcast (B, 1, 1, 1, 1) against (B, K, C, H, W) as before. gm_mean, the gm_mode "mean" output, and diagnostics (n_eff mean over (1, 2), max_w mean over (0, 2, 3)) work on (B, K, 1, 1, 1) and give per-image values.
4. **fp32 precision**: the fp32/fp64 relative error of the loss was ≤ 1.1e-6 at sigma_high 0.9 to 1e-4. The absolute error of the summed log-likelihood (0.01-0.47 nats at sigma ≥ 0.1, 65 nats at 0.01 against a gap of 3.4e4) is far below the top-2 gaps, so the winners and the logsumexp are right. The logits are cast to fp32 before the mean over 65536 pixels (`JiT_T2I.py:580`).
5. **Model, EMA, checkpoints**: no new parameter (T32 checks the state_dict keys and values). ema_denoiser is a deepcopy, so it keeps gm_scope. LightningModel does not save hyperparameters, so LightningCLI's ckpt hparam parse does nothing. eval_gmflow.sh passes `--model.denoiser.init_args.gm_scope image` to predict, so euler_mean (mean mode) and gm_draw both use image weights. The shape assert in the eval heredoc (8 x 4 x 256 = 8192 output rows) still holds. scope.txt guards a scope mismatch (T32 and the smoke check the messages).
6. **torch.compile and DDP**: only the denoiser is compiled. gm_scope is a constant attribute, so the mean over pixels compiles into the graph with no new guard. The trainer's image_scope() checks are Python shape compares, run eagerly, with no sync. Every parameter stays in the graph (the losers' rows get exact zeros), so DDP sees no unused parameters. On steps that write no logs, nothing new runs: win_stats runs only when diag_now or GPIC_GM_PRINT is set.
7. **Sampler** (`sampling_gmflow.py:114-121`): the point-model and per-pixel branches are unchanged. The image branch picks one k per image (Gumbel-max on (B, K)) and takes the straight step to mu_k with no noise; the last step returns mu_k. The per-rank seed in `_start` gives each rank its own draws. gm_draw and gm_convert assert against image weights, so gm_sde, gm_ode and order 2 cannot run silently (eval_gmflow.sh also refuses them first).
8. **Unchanged behaviour of existing runs and queued jobs**: with no `--scope`, train_gmflow.sh builds the same command line (T32 checks "gm_scope" is absent), and EV_OPTS is unchanged. evals_gmflow.sh defaults to pixel. eval_gmflow.sh defaults EVAL_SCOPE to pixel, adds no argument for pixel, and a root without scope.txt is pixel. Every queued gmflow wq job I listed (w1185 … w1289) runs from a frozen worktree (for example w1185: `gpic-wt/gmflow_run2/jobs/train_gmflow.sh`), so this uncommitted change cannot reach them. A run root that already has last.ckpt and no scope.txt is recorded as pixel and refuses `--scope image` (T32).
9. **Fair comparison with w1173**: w1173 = `run_gmflow.sh` at df37cea, `GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0`, 62336 steps, ms at 15584/31168/46752/62336, gm_k 8, lam 0.5. The planned run uses data w1147 (the same two env values), 62336 steps (ms at 31168 = C0/16 and 62336 = C0/8), and the same config overlay. I diffed df37cea..HEAD on the training paths: JiT_T2I (the gm_spread "image" else-branch is the old code), training_gmflow (prints only), lightning_model_gpic (pre-clip norm now logged for every model; GPIC_GRAD_PRINT unset). None of these changes the training math. The only difference is the scope.
10. **Option parsing**: the planned order `--wt gmflow_run6 --scope image` matches the fixed parse order (--wt, --spread, --lam, --scope). `--scope image` with the default spread passes the spread check. EV_OPTS becomes `--wt gmflow_run6 --scope image`, which matches evals_gmflow.sh's order (--link, --wt, --spread, --table, --noise, --scope).
11. **Smoke script regexes and paths**:
    - The trainer print (`training_gmflow.py:61`, image spread) is `loss %.5f s_u %.4f n_eff %.3f max_w %.3f win_max %.3f win_alive %d`. The smoke's regex matches it field by field, and "nan" / "inf" are matched and then fail isfinite.
    - The grad print (`lightning_model_gpic.py:141`, `[gmflow] call N: grad norm before the clip X`), `[warmup] done` (compile_warmup.py:52) and `resuming from …last.ckpt` (train_ladder.sh:34) match.
    - train_ladder 1600 steps gives ms every 400 and last every 50, so ms-step=000400 exists and the resume starts from step 400.
    - The saved config holds `gm_scope: image`, in the same place as w1173's `gm_k: 8`.
    - The eval output dirs `eval_000400_<S>_calls<C>_official` match eval_dir with the default official noise.
    - Both guards exit 2 with the grepped messages before any model loads.
    - EVAL_SCOPE is set only per command, so the w1173 regression eval runs as pixel. Its root has no scope.txt, and the symlinked checkpoint is found by `ls -t ms-*062336*`.
    - The script returns `fail` (0 or 1), and exits early with the training's rc if the first training fails.

## Summary

The code is correct, and the comparison with w1173 is clean. Before the 8-GPU run, add `gm/w_dead` (S1, about 3 lines plus the smoke regex) so that a collapse shows in the first logs. Record S2 (the weights learn about 2000x slower) in section 15 before reading A or w_dead. M1, M2 and M5 take a few minutes each. M3, M4 and M6 need only notes.
