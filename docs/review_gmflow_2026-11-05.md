VERDICT: GO — the loss, the samplers and the configs match GMFlow and w1147; fix findings 1 to 3 (they are about the protocol, not the math, and each is cheap) before you submit the real runs.

Review of branch `gmflow`, commits 9711ff9..4c22ff5, worktree `gpic-wt/gmflow`. The reviewer did not edit the code, commit, or submit any job.
My own check scripts are in `/scratch/gpfs/GROUP/USER/tmp/review_gmflow/` (`port.py`, `chk.py`, `cfg.sh`, and the resolved configs `k8.yaml`, `k1.yaml`).

## Findings, most severe first

### 1. The comparison line for each sampler and call count is not fixed (HIGH, readability)

- Where: `docs/gmflow.md:20-27` (plan item 4, owner decisions 2 to 4). `jobs/run_gmflow.sh:37-46`, `jobs/eval_point_sampler.sh:16`.
- Failure scenario: each call count gives 3 GMFlow numbers (GM-SDE, GM-ODE, Euler on the mean) and up to 3 point-model numbers (official Euler, DDPM-small, exact DDIM). At 50 calls these are 1539.52 / 1089.92 (Euler), 1566.92 / 1142.48 (exact), and DDPM-small, which is not measured yet. At 8 calls they are 1709.60 / 1248.78 (exact), plus Euler and DDPM-small, which are not measured yet. The plan names only 1539.52 and 1089.92. It does not say which point-model sampler sets the line at 8 calls, or whether GMFlow is judged on GM-SDE only or on its best sampler. Nine pairs are possible at each call count. If the pair is picked after the results, the 5% rule means nothing. The energy review (F1) found the same confound and fixed it before the run.
- The same gap exists for K = 8 against K = 1 (decision 1). No rule says what result shows that "the mixture" mattered.
- Fix: before you submit, write the line into `docs/gmflow.md`. Suggested rule: at C calls and each compute point, line = 0.95 × the lowest point-model FD over the samplers that point model has at C calls. This is the strongest fair baseline. The verdict reads GM-SDE (the approved main reading). GM-ODE and Euler on the mean are reported but not judged. For the mixture claim: K = 8 GM-SDE must be 5% below K = 1 GM-SDE at both compute points.

### 2. The jobs run from the live development worktree, not from a frozen one (MEDIUM)

- Where: `jobs/run_gmflow.sh:20-23`, `jobs/eval_gmflow.sh:19,32`, `jobs/eval_point_sampler.sh:12-14`. Every script uses `WT=$WC/project/gpic-wt/gmflow`.
- Failure scenario: the guard `git status --porcelain --untracked-files=no` runs once, at job start, and finds only uncommitted edits. Suppose a commit lands on branch `gmflow` while a run is in flight or queued. Examples: a fix after smoke w1168, or a commit of this review. Then each of the 12 evals starts a new `python main.py` that reads the new code. K = 8 and K = 1 can also train on different commits. `launches.txt` and `ckpt.txt` record the sha, but nothing prevents the mismatch. Tests B, C and E each used a frozen worktree for exactly this reason (commit 9a9e72d: "frozen worktrees of their own"; `run_energy.sh` uses `gpic-wt/energy_real`).
- Fix: after the smoke test passes, run `git worktree add --detach gpic-wt/gmflow_real <sha>`. Point `WT` in all three scripts at it, and add a guard `[ "$(git -C $WT rev-parse HEAD)" = <sha> ]`.

### 3. The main reading, GM-SDE, has no final-mean variant in the real run (MEDIUM)

- Where: `jobs/run_gmflow.sh:39` loops over `gm_sde gm_ode euler_mean` only. The report (`docs/gmflow_impl_report_2026-11-05.md`, "Open risks" 1) and `docs/gmflow.md` 4.5.1 name `gm_sde_final_mean` as the check for per-pixel independent draws, but no job runs it.
- Failure scenario: at 8 calls the last GM-SDE step returns x̂ = μ_k + s_x ξ, with s_x = s_u / 8. The noise is independent at each of the 256² × 3 values. This is the paper's ablation A2: first order with no spectral sampling, FID 8.98 against 3.43 for the full sampler (digest section 8). Suppose GM-SDE loses to the point model. The run then cannot say whether the head failed or the last per-pixel draw added noise texture that FD-DINOv2 penalises. The brief (J3) lists `gm_sde_final_mean` as an eval sampler, so the script already supports it.
- Fix: add `gm_sde_final_mean` at 50 and 8 calls at both compute points. That is 4 more evals, about 4.5 min each (w1147's evals: predict 145-205 s plus FD 74 s). Either raise `--time` to 03:00, which is still within the 3 h limit for jobs that are not resumable, or queue the 4 evals as separate `eval_gmflow.sh` jobs after training.

### 4. How often the gradient clip at 1.0 acts is unknown for both arms and for w1147 (MEDIUM, not blocking)

- Where: `configs_h100/lite_ladder.yaml:88-89`; `src/lightning_model_gpic.py:134-136` logs the pre-clip norm only when `gm_k > 0`.
- Failure scenario: I computed the gradient on a mean element under the λ = 0.5 transition loss, as a ratio to the point model's v-loss gradient (`chk.py`; sum over 3 channels × 2 against the mean over 3 channels, divided by σ²). The ratio is 1.6, 2.0 and 2.2 at σ = 0.9, 0.5 and 0.05 when s_u = 1 (the init). It is 3.4 to 7.8 when s_u = 0.2, and 3.6 to 8.6 when s_u = 0.05. The GMFlow residual also carries the irreducible transition noise √c3. The logits and the s-MLP add their own gradients. So GMFlow's pre-clip norm is likely 2 to 8 times that of w1147. If w1147 sat near or below 1, GMFlow is clipped on most steps and w1147 is not. Adam removes a constant scale. But a clip that acts on most steps rescales each batch by a different factor, so batches with high-loss t lose weight. That is a difference between the arms that nobody controls or measures. w1147 logged no gradient norm.
- Fix: read "share above the clip 1.0" in the smoke log of w1168. Remove the `gm_k` gate at `lightning_model_gpic.py:135`, so that any short point-model run logs the same quantity. Report both shares next to the result. Do not change the clip: parity with w1147 is the design.

### 5. The init of the K = 1 control differs from K = 8 and from the point model (LOW)

- Where: `src/models/transformer/JiT_T2I.py:443-448`. The final Linear keeps `_basic_init`'s Xavier-uniform weight for every `gm_k`.
- Failure scenario: Xavier's std depends on fan-out. For unit LayerNorm inputs, the initial mean output has std 0.38 at K = 8 (fan-out 8192) and 0.88 at K = 1 (fan-out 1024). The point model starts at exactly 0. So K = 1 against the point model differs in loss and in init. K = 8 against K = 1 differs in K and in the initial output scale. A std of 0.88 on a [-1, 1] image is a large random output at step 0. The 4000-step warmup probably washes it out, but it is a third variable in the control that is meant to isolate one cause.
- Fix: K = 1 has no symmetry to break, so zero-init its mean rows as the point model does. If you want the same init rule in both arms, scale the Xavier weight so that the per-output std is the same at K = 1 and K = 8. If you change nothing, say so in the result.

### 6. `eval_point_sampler.sh` writes into w1147's run directory (LOW)

- Where: `jobs/eval_point_sampler.sh:14,19` set `LADDER_RUN_ROOT=$WC/runs/gpic-speed/point_nodrop_d640`, so `eval_gmflow.sh:33` creates `point_nodrop_d640/exp_ladder_d640/eval_<step>_<sampler>_calls<C>/`.
- Failure scenario: 6 new directories appear next to w1147's checkpoints. The names do not collide with the existing `eval_*_prompt` directories, and the metrics.json guard prevents overwrites, so no result is lost. But the brief asked for no writes there, and the smoke test (`jobs/smoke_gmflow.sh:66-68`) shows the clean way: a symlink in a separate root.
- Fix: use a root `runs/gpic-speed/point_nodrop_d640_samplers/exp_ladder_d640/` that holds symlinks to `ms-step=033408.ckpt` and `ms-step=066816.ckpt`. Put the skip check (`eval_point_sampler.sh:19`) on that root too.

### 7. A resumed run reads a different data stream from w1147 (LOW, only if the run is cut)

- Where: `src/data/dataset/gpic_dataset.py:48-63` (`EpochShuffledShards` restarts `passes` at 0 in every process), `jobs/run_gmflow.sh:27` (fixed `GPIC_SHARD_SEED`), `jobs/train_ladder.sh:29-31` (resume from `last.ckpt`).
- Failure scenario: w1147 trained in one launch (`point_nodrop_d640/exp_ladder_d640/launches.txt` has one line). Suppose run_gmflow is killed (node fault, `exit 98` from the health gate, a timeout) and rerun. The resumed run then starts the same shard permutation from its beginning, so the arm reads the early shards twice and never reaches some later ones. The arm then sees different data from the control.
- Fix: nothing to do if the run finishes in one launch. Check that `gmflow_k{8,1}_d640/exp_ladder_d640/launches.txt` has one line, and flag the result if it has more.

### 8. The compute will likely exceed the owner's estimate (LOW)

- Where: `docs/gmflow.md:27` (decision 5: about 25 H100-h in total).
- Estimate: each arm trains for about 40 to 45 min. That is w1147's 36 min, plus 7% more FLOPs, plus the fp32 loss on (32, 8, 3, 256, 256) tensors in eager mode. Each arm then runs 12 evals of about 4.5 min. So one arm is about 1.6 h × 8 GPUs ≈ 13 H100-h. Two arms make 26 H100-h. The point-model sampler controls add about 3 H100-h and the smoke test about 7. The total is about 36 H100-h, about 45% over the estimate, before finding 3 adds 2 to 3 H100-h.
- Fix: tell the owner the revised number before you submit.

## Checked and correct

- **Config parity (Q1).** I resolved the real K = 8 and K = 1 command lines with `main.py fit --print_config` (`cfg.sh`) and diffed them against the config that w1147 saved (`point_nodrop_d640/exp_ladder_d640/config-TrainerFn.FITTING-2611050106.yaml`). The only differences are: the logger name and id; `max_steps`; the ms cadence (15584 or 16640) and the `last` cadence (1948 or 2080); `default_root_dir`; `gm_k` (plus `head_width: 0`); and the trainer class with `lam: 0.5` and `weight_scale: 2.0`. All of these are the same: `null_condition_p: 0.0`, `seed_everything: 0`, `gradient_clip_val: 1.0` with `norm`, `train_batch_size: 32` on 8 GPUs, AdamW (lr 1e-4, betas 0.9/0.95, wd 0, fused), warmup 4000 then flat, EMA 0.9999, `lognorm_t` with P_mean -0.8 and P_std 0.8, timeshift 1, bf16-mixed, the DDP settings, and the data. The job environment is the same too: `GPIC_SHARD_SEED=1730000000` and `GPIC_REGION_CROP=0` (`run_gmflow.sh:27` = `ar_real/jobs/run_point_nodrop.sh`), and the same `train_ladder.sh` callbacks (CompileWarmup, HostTuning, ms first).
- **Shared code, ar_real@9a9e72d against gmflow HEAD, outside the GMFlow paths.**
  - `lite_ladder.yaml:154`: `null_condition_p` is 0.0 at 9a9e72d and 0.1 at HEAD, because the gmflow branch forked before 2ce6bd6. `gmflow_train.yaml:18` sets 0.0, and the resolved config confirms 0.0.
  - `JiT_T2I.py`: EnergyHead, which is inert at `head_width: 0`. The `ypos` dtype refactor gives the same value. The gm_k = 0 init is the same zero weight and zero bias. T7 checks bit-identity against 9711ff9.
  - `lightning_model_gpic.py`: `_log_now` is the same condition, now computed once; the predict wrapper only forwards `**kw`.
  - `training_implicit.py`, `training_ar.py`, `AR_JiT.py` and `jobs/eval_ladder.sh` differ, but neither the GMFlow path nor the point model's eval uses them.
  - `sampling.py`, `src/data`, `src/callbacks` and `main.py` are unchanged.
- **Loss (Q2).** I wrote an independent port of GMFlow's u-branch (`gmflow.py:176-204` + `diffusion_loss.py:33-36`, in GMFlow's own parameterisation) and compared it with `transition_nll` at t = 0.1, 0.5 and 0.93, K = 4. The largest difference is 1.5e-5 (fp32). The values match:
  - c1, c2 and c3 (`gmflow.py:12-21`);
  - the forward transition: the marginal of x_high is (1 − σ_h) x + σ_h ε (variance m²σ_l² + σ_h² − m²σ_l² = σ_h²);
  - s_x = σ_high · s_u (`gmflow.py:31-33`, `training_gmflow.py:46`);
  - the sum over the 3 channels, the mean over pixels, × 2, and the mean over the batch (`training_gmflow.py:46-47`);
  - the inverse-std clamp at 1e4 and σ ≥ 1e-4.
  The clamp on c3 is 0 instead of 1e-6. It differs from GMFlow only when c3 < 1e-6, that is t > 0.998, and it gives the exact λ = 1 reduction (T3). The s-MLP input is detached (`JiT_T2I.py:528-530`). Its last layer is zero-initialised, so s_u = 1 at init (T8).
- **The CPU tests.** I reran `tools/gmflow/test_gmflow.py` on vis1: ALL PASS. T1's ratio is 6.122448 against 6.122449.
- **GM-ODE (Q3).** I wrote an independent port of the loop at `gmflow.py:527-539`, `denoising_gm_convert_to_mean_jit` (`:44-70`) and `FlowEulerODEScheduler.step`. Against `GMODESampler`, the largest difference is 6e-8 at 8 × 16 and 9e-8 at 5 × 3 (`port.py`). `x_t_base` and `t_base` (= x_src and σ_src) stay fixed for each call, as in the reference. The sub-step grid is uniform with N·n points, and the last sub-step lands on the reweighted mean.
- **GM-SDE.** The Gumbel term −log(−log u) is correct, with u in [1e-12, 1 − 1e-7]. The noise s_x ξ goes on all 3 channels. On the first step (σ = 1), c1 = 0, c2 = α_to and c3 = σ_to². On the last step, c1 = 0, c2 = 1 and c3 = 0. The point model (gm_k = 0) is DDPM with the small variance (`sampling_gmflow.py:59-60`). Euler on the mean is `EulerSamplerJiT` unchanged on the `gm_mode="mean"` output. CFG = 1.0 is asserted. The per-rank seed is set once (`sampling_gmflow.py:31-33`).
- **FLOPs and steps (Q4).**
  - 62336 × 1.07010 = 66706 ≤ 66816, and 66560 × 1.00245 = 66723 ≤ 66816. Both values are multiples of 128 and of 32 (as `train_ladder.sh:22` requires).
  - The ms cadences STEPS / 4 give checkpoints at 31168 and 33280. 31168 × 1.0701 = 33353, against 33408 for w1147.
  - The LR is flat after warmup, so the length of the schedule is not a confound.
  - K = 8 sees 6.7% fewer images. That is the iso-FLOP design, as in the energy run.
- **DDP and compile (Q7).** K = 1: log_softmax over one element has an exact zero gradient, so its logit rows of the shared Linear get zero gradients. DDP sees no unused parameter. The s-MLP runs in every training forward (`gm_mode="full"`, also in CompileWarmup through the trainer). The EMA is a `deepcopy` of the denoiser, so it holds the s-MLP (`lightning_model_gpic.py:66`). `gm_mode` is a constant Python string, so it adds one graph in predict and no recompiles in training. The diagnostics run outside the compiled denoiser. Logging from `configure_gradient_clipping` is allowed (Lightning `fx_validator.py:46`). The checkpoint guard checks the head size, the presence of the s-MLP and the width (`eval_gmflow.sh:38-47`).
- **Numerics (Q6).**
  - One fp32 (32, 8, 3, 256, 256) tensor is 201 MB. Autograd keeps about 10 of them, about 2 GB, which is small on 80 GB.
  - The t sampler gives P(t > 0.9) = 9e-5, about 1400 samples over the run, and P(t > 0.95) = 1.4e-6, about 23 samples, so the 7e7-scale losses of T8 are rare and the clip bounds them.
  - The x-means come out of a bf16 Linear, with a quantisation error of about 0.002. This matters against the transition std 0.44 σ_h only when σ_h < 0.01, where P ≈ 1e-11.
- **Job scripts (Q5).**
  - `#FETCHABLE`, `--comment=fetchable`, `-c 96` and `--mem=800G` are present. TMPDIR and `module load proxy/default` come from `env.sh`. The single argument K reaches the script through wq's args file.
  - 02:45 is within the 3 h limit for jobs that are not resumable. A rerun resumes training from `last.ckpt` and skips evals that have `metrics.json`. The eval output name in `run_gmflow.sh:40` matches `eval_gmflow.sh:33`.
  - The 50K captions come from `gpic_test_50k.jsonl`, with `noise_per_prompt true`, EMA weights (`eval_original_model: false`) and no overwrite.
  - Time: training fails to fit in 02:45 only if s/step exceeds about 0.10 s, which is 3 times w1147's 0.032 s. The smoke test reports s/step.

## Not checked

- Anything on a GPU: compile with DDP, recompiles, resume, predict through `main.py`, step time and memory. Smoke w1168 covers these; read its `phases.log`, especially s/step, the recompiles after warmup and the share of clipped steps.
- The FD pipeline (`tools/fast_eval.py`), and FD variance across seeds. No second seed is planned, so a margin near 5% is not resolvable.
- Whether the s-MLP's input (t embedding plus the masked mean of the y_embedder output) is the best choice. It follows the brief.
- Spectral sampling and the second-order solvers. They are out of scope by design.
- The reference repository was only read, never run.
