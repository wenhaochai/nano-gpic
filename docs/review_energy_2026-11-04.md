VERDICT: NO-GO. The code is correct, but the reading rule is not. At N = 256 the exact posterior is a single image at 99.99% of training t. So collapse rule 7.4 will flag a correct model, and the job cannot tell the head from the point model. Fix the doc (finding 1) and add a probe at small t (finding 2), then run the job unchanged.

# Review of branch energy-head (e929fb6, 49be9a7), 2026-11-04

## Findings, most severe first

1. **docs/energy_head.md:160 (rule 7.4, collapse) measures the wrong thing.**
   - I computed the exact posterior over the 256 training images. Its effective number of images (exp entropy), averaged over 64 x_t from training images, is:
     - about 256 at t = 0
     - about 100 at t = 0.005
     - about 17 at t = 0.01
     - 1.19 at t = 0.02
     - 1.01 at t = 0.03
     - 1.00 at t ≥ 0.05
   - Training draws t = sigmoid(N(-0.8, 0.8)). So 0.006% of training t is below 0.02, and 0.04% is below 0.03. In 24000 steps × 32, that is about 43 draws below 0.02.
   - When the posterior is one image, an exact sampler gives between = 0, and to_data = 0. So the ratio between / to_data is 0/0. In practice it measures the model's error spread, not collapse. A correct head should drive between toward 0 at nearly all training t. Rule 7.4 will then read "collapsed".
   - The ratio is also an EMA over steps with mixed t. It cannot be read per t.
   - The claim "the ratio is 1 for an exact sample" holds only where the posterior has spread. Here that is t < 0.02.
   - Fix: delete rule 7.4. Replace it with a per-t diagnostic after training: at fixed t ∈ {0, 0.005, 0.01, 0.02, 0.05}, draw 64 samples per x_t. Compare the number of distinct nearest images against the exact posterior above.

2. **The test is barely informative about the head (question 4).**
   - The image is chosen at t < 0.02 for both arms. The networks almost never train there.
   - At 50 calls the grid is 0, 0.02, 0.04, …. So the choice of image happens in the first one or two calls. Both models then pick the image from the initial noise.
   - Expected result: covered, distinct and effective_images tie with the point model, within its range (0.81 to 0.88 covered).
   - The head is not blind at t = 0. The backbone sees x_t = ε, so its features carry randomness. A head could map them to 256 different images, as a one-step generator does. The point model cannot, under MSE.
   - The training t distribution never asks for this, so the head will not learn it.
   - Two outputs can show a real effect:
     - `one_call_near`: about 0 for the point model.
     - The 8-call metrics, whose grid is 0, 0.125, ….
   - Fix (cheap, optional): add to the energy arms only a share of t drawn from U[0, 0.05] (for example 10%), or report the per-t probe from finding 1. Without it, a tie means "the head was not asked", not "the head cannot".

3. **docs/energy_head.md:152-158 (rule 7.2) and the λ choice (question 4).**
   - Training t carries a one-image posterior almost everywhere. There, λ = 1 trains the head to output a spread equal to its own error, and the sampler adds that spread at every call. Over 50 calls this is the variance build-up that De Bortoli et al. report for λ = 1 at many steps.
   - So expect near and nn_rms at 50 calls to be slightly worse than the point model. Rule 7.2.2 may then fail for that reason alone.
   - Fix: replace the implicit arm (reference only, +32% FLOPs, the slowest arm) with energy λ = 0.5, seed 0. This needs a small code change. `training_implicit.py:39` hard-codes 0.5 · between (λ = 1). Add a `lam` argument, with default 1. I made no edit.

4. **tools/energy/overfit_energy.py:127-130: the control does not replay w1095's RNG stream after step 1000.**
   - Seed, data, model init and the first 1000 steps match w1095. I checked: same seed, same image load, the same init stream (test 1), and JiTTrainer draws the same as CodeJiTTrainer at R = 0.
   - The extra 8-call sample at each eval uses the global RNG. So the batches, t and noise after step 1000 differ from w1095. torch.compile is also not bitwise deterministic.
   - So covered 0.81 will not reproduce exactly. Read the control as a third seed, against the range 0.81 to 0.88. Only a gap well outside that range points to a bug.
   - Fix (optional): run the eval sampling under `torch.random.fork_rng()`.

5. **docs/energy_head.md:158 (rule 7.2.4, effective_images ≥ 1.1 × control) is weak.**
   - Seed noise is about 5% per run. The difference of two runs then has about 7% noise, so 1.1 is about 1.4 σ.
   - Both energy seeds compare against the same single control run, so the two checks are correlated.
   - It still passes the false positive rate as a joint rule with 7.2.1, but on its own it is not above noise.
   - Fix: keep it, but use it only as a joint condition. Weights of w1095 and w1113 were not saved, so no other control values exist.
   - Rule 7.2.1 (covered above the maximum of 3 point runs, on both seeds) is fair. If the energy head equals the point model, the chance that both of its seeds rank above all 3 point runs is 1/10.

6. **jobs/overfit_energy.sh:33 and line 5: the time margin is thin but enough.**
   - From w1113: 515 s for the point arm. With an A40 sample call of 20 ms, I estimate about 15.6 ms per point train step on H100.
   - Implicit arm: 24000 × 20.3 ms ≈ 490 s. Sampling: 6 evals × 64 batches × 58 calls ≈ 22k calls × 7 ms ≈ 155 s. Compile: about 60 to 120 s. Total: about 700 to 770 s.
   - That is under the 1000 s timeout. The worst arm has about 25% margin.
   - The 8-call sample costs about 6 × 64 × 8 × 7 ms ≈ 22 s. The 1-call sample costs under 1 s.
   - The job limit is 1080 s. timeout (1000) + kill (60) + gpu_health leaves under 20 s. If an arm hits its timeout, the job can hit the limit before the tail prints.
   - Fix: set the timeout to 950, or `--time` to 00:20:00.
   - The point arm ends about 3 to 4 min before the implicit arm. GPU idle time is therefore minutes, which is fine.

7. **Minor: question 4 states about 250K parameters per head block.** A ResBlock at W = 352 has 248K in the MLP plus 372K in adaLN, so about 620K per block. The conclusion does not change.

## Checked and correct

1. Pairing (question 1). EnergyHead uses `repeat_interleave(m, 0)`, so the order is [a, a, b, b]. The trainer uses `g.view(B, m, ...)`, which takes the same order.
   - With the noise layers set to 0, the m = 2 output equals the m = 1 output of the same input, sample by sample. Max error: 0.0.
   - A tag network that outputs its input index gives to_data = 1e-4 = sqrt(1e-8), which is 0 within the epsilon. So x[k] is paired with g[k, :].
   - The implicit model's fork in the backbone uses the same order.
2. The backbone runs once per x_t. The head runs at batch 2B. The test hooks confirm this.
3. The noise is fresh at each call, also under torch.compile. On CPU, two compiled calls on the same input differ by 0.27, and the two m = 2 samples differ by 0.26.
4. Sampler at m = 1 (question 2).
   - It uses `net(x, t, y, y_mask=mask)`, so m = 1 by default.
   - The exact step uses no t_eps. The largest grid point is 0.98, so it never divides by 0.
   - It matches `ImplicitSampler._impl_sampling` line by line.
5. The point arm is the official model, JiTTrainer, and the Euler step with t_eps = 0.05, as in overfit_code.py. Seed 0 and the tag `_s0` are kept apart from w1095's files.
6. Metrics (question 3).
   - nn_rms, near, covered, distinct and pixel_std are copied from overfit_code.py.
   - effective_images is exp(entropy) of the share of samples per nearest image, with no threshold, over 8N = 2048 samples. This is the same as `effective()` in tools/ar/eval_eta.py.
7. `tools/energy/test_energy.py` passes on vis2 CPU. The FLOPs check measures what it states: the difference of 674.6 MFLOP equals the hand formula.
8. The job script:
   - It checks the worktree for local changes and old code, and checks GPU health.
   - It gives one GPU per arm, with `CUDA_VISIBLE_DEVICES` respected for a guest.
   - It carries `#FETCHABLE` and `--comment=fetchable`.
   - env.sh sets TMPDIR, the proxy and the venv.

## Not checked

1. No run on a GPU. No H100 timing. Timings are estimates.
2. bf16 autocast effects on the head's noise, which is drawn in bf16 because `h.dtype` under autocast is bf16. Here it is only a source of randomness, so the effect is probably harmless.
3. Whether the head trains stably over 24000 steps.
4. The papers' claims. I took section 5 of the design doc as given.

## 修复（2026-11-04）

1. 发现 1：删掉判据 7.4，加了分布探针，读法写明精确后验的有效图数。
2. 发现 2：加了一个 10% 的图 t 在 U[0, 0.05] 的 arm。打平的读法写明“没有问到”。
3. 发现 3：隐式 arm 换成 λ = 0.5，训练器加了 `lam`。
4. 发现 4：读法写明控制组当第 3 个种子读。没有用 fork_rng。
5. 发现 5：写明只作联合条件。
6. 发现 6：timeout 改成 950。
