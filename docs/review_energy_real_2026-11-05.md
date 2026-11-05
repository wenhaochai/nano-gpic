VERDICT: GO. Nothing blocks the smoke test or the run. One extra evaluation arm (F1) is needed before section 9 can be read.

Review of 74d50bb (branch energy-head). No code was run on a GPU. This review read the code and compared it with the control w1147 (`../ar_real`, commit 9a9e72d).

## Findings, most severe first

F1. The sampler confound is real, and section 9 gives no fix for it. `docs/energy_head.md:209` (rule 4); `src/diffusion/flow_matching/sampling.py:207-219`.
- Failure: the control's Euler clamps 1 - t at t_eps = 0.05. At 50 calls the timesteps are 0, 0.02, ..., 0.98, so the clamp acts at t = 0.96 and t = 0.98. At the last step x = x + 0.4 (hat - x), so noise stays in the image (0.0144 of the range, per the `land` comment). ImplicitSampler lands on hat exactly. So the 50-call FD differs by construction, and the 5 % threshold mixes the sampler with the model. At 8 calls the clamp acts only on the last step (t = 0.875 < 0.95), but the final noise still differs.
- Fix: add one arm. Evaluate w1147's `ms-step=033408` and `ms-step=066816` with the ImplicitSampler block of `energy_sample.yaml`, but without `head_width`, at 50 and 8 calls, with `noise_per_prompt true`. For a point model this sampler is exact DDIM: x' = t' hat + (1 - t') eps_hat. The arm needs a copy of eval_energy.sh without the `xi_global` assert. It costs about 4 x 5 min. Read rule 1 against this arm. Report the Euler numbers next to it. Note: `eta: 0` does not give this sampler, because `if self.eta:` treats 0 as off. `land: true` fixes only the last step.

F2. All 8 ranks draw the same t, noise and head noise ξ at each step. `src/callbacks/host_tuning.py:118`; `jobs/env.sh` does not set `GPIC_RANK_SEED`.
- Failure: a global batch of 256 has only 32 distinct (t, noise, ξ). This is the same for w1147, so the comparison stays fair. The energy score's two samples per image still differ. Not a blocker.
- Fix: none for this run; changing it breaks the match with the control. Mention it in the report.

F3. In predict, ξ comes from the global RNG, and every rank starts from `seed_everything: 0`. `src/models/transformer/JiT_T2I.py:258-259`.
- Failure: rank r and rank r' use the same ξ for batch i. The prompts and x_T differ, so FD is still valid. But an image's ξ depends on its rank and batch position, not on its prompt. The same prompt gives another image with another number of GPUs.
- Fix: none needed for this run. Write it down as a limit of reproducibility.

F4. The time notes in the brief are pessimistic. Not a failure.
- w1147 trained 66816 steps in about 36 min (01:06 to 01:42), so about 32 ms per step. Its 50-call predict took 205 s and 193 s; fast_eval took 75 s. The energy head at +10 % wall time (A40, section 2) needs about 40 min to train. Four evaluations need about 4 x 4.5 min. Total about 65 min against 2 h. The smoke test (600 + 420 + 900 s caps, 32 min) fits in 50 min; the earlier d640 smoke trained 600 steps in 5 min.

## Checked and correct

Question 1, paths and control.
- `smoke_energy.sh`, `run_energy.sh` and `eval_energy.sh` all use `gpic-wt/energy_real`. No other path assumes another worktree. Logs and `env.sh` come from `project/gpic`, as in the control.
- `jobs/train_ladder.sh` is byte-identical to `../ar_real`. It honours `LADDER_WT`, `GPIC_REGION_CROP` and `GPIC_SHARD_SEED` the same way. `tools/fast_eval.py` and `tools/gpu_health.py` are identical too.
- `git log f429d51..ar`: the commits after 9a9e72d change only `docs/autoregressive.md`, `jobs/run_ar.sh` and `jobs/run_ar_few.sh`. Nothing touches the comparison.
- Diff 9a9e72d..HEAD in shared code: `lite_ladder.yaml` only `null_condition_p` 0.0 → 0.1; `JiT_T2I.py` only code behind `head_width`; `training_implicit.py` only `lam` and `low_t_p` (off). `energy_train.yaml` replaces the whole trainer with `null_condition_p: 0.0`, and your `--print_config` confirms it.
- Same `GPIC_SHARD_SEED=1730000000`, `GPIC_REGION_CROP=0`, d640/9/2, and `noise_per_prompt true` as the control's `EVAL_NOISE=prompt`.

Question 2, DDP and Lightning.
- With `head_width`, `final_layer` is an `EnergyHead`; no unused `FinalLayer` exists. Every head parameter (`norm` has none, `inp`, `cond`, `xi_global`, `xi_token`, 2 ResBlocks, `out`) is used on every m = 2 step. The backbone has no `xi_*` layers because the assert forbids `xi_dim` with `head_width`.
- `lightning_model_gpic.py` and the callbacks never branch on `xi_dim`, `m` or `is_implicit`. Branches exist only for `is_ar` and `code_dim` (getattr default 0).
- `CompileWarmup` calls `pl_module.diffusion_trainer(...)`, so it runs the same `_impl_trainstep` with `m=self.samples` = 2, inside DDP's forward context. No m = 1 training graph exists, so no recompile per step from m. The smoke test counts recompiles after `[warmup] done`.
- Trainer output keys: `loss`, `to_data`, `between`, `loss_per_sample`. `training_step` pops `loss_per_sample` and logs the rest. The gradient clip path skips the `code_dim` print.

Question 3, evaluation.
- `ImplicitSampler` calls `net(x, t, condition)`, so m = 1. `predict_step` wraps net only for `y_mask`; that wrapper passes no `m`, so m = 1. CFG is asserted 1.0.
- The EMA net is used (`eval_original_model: false`), cast to fp32 at `on_predict_start`.
- `noise_per_prompt` seeds x_T with `noise_seed + idx`, as in the control.
- MS = STEPS / 4: real run 16384, so ms checkpoints at 16384, 32768, 49152, 65536. The smoke test with STEPS = 3200 gives MS = 800 and LAST = 100, so `ms-step=000800.ckpt` exists after the first part; the resume starts from `last.ckpt` at 800. The eval glob `ms-*000800*` matches.
- The guard on `ema_denoiser.final_layer.xi_global` holds for this model. Output goes to `eval_<step>_calls<N>_prompt`, one per call count; it does not overwrite.

Question 4, steps.
- 66816 / 1.0194 = 65545.4; 65536 is 2048 x 32. Energy FLOPs at 65536 are 0.014 % below the control. 32768 x 1.0194 = 33404 vs 33408. Both are fair.
- The 1.94 % is per forward, with the head at m = 2 for the per-token parts and m = 1 for the projections, measured with `FlopCounterMode`. Training counts forward x 3 on both arms, so the ratio carries over.

Question 6, EMA.
- `ema_denoiser = copy.deepcopy(self.denoiser)` comes before compile, so it holds the head. `SimpleEMA` lerps `list(net.parameters())` against `list(ema_net.parameters())`; the order matches. The head has no buffers.
- ξ is drawn in `h.dtype`. Under bf16 autocast, `h` after the Linear is bf16 in both training (fp32 weights) and predict (fp32 EMA weights). So ξ has the same dtype and law in both.

Section 9.
- Thresholds: 1539.52 x 0.95 = 1462.5; 1089.92 x 0.95 = 1035.4. Correct.
- Rule 4 is right: ImplicitSampler has no t_eps, and its last step lands on hat. It is incomplete without F1.

## Not checked

- No GPU run. The smoke test will be the first check of Lightning compile, DDP, resume and predict with this model.
- The wall-time cost of the head on H100. Only the A40 number exists.
- Whether `CompileWarmup` with B = 32 compiles the head's `repeat_interleave` with a dynamic batch that the real per-GPU batch of 32 hits. It should, because the batch sizes match, but this was not verified.
- `jsonargparse` merging with `--print_config` beyond what you ran.
- fast_eval internals and the reference stats.

## 修复（2026-11-05）

1. F1：加了 `jobs/eval_point_exact.sh` 和 `configs_h100/exact_sample.yaml`，第 9 节读法 4 改了。`eval_energy.sh` 的断言改成检查点和配置必须一致。
2. F2、F3：写进了第 9 节读法 5。

