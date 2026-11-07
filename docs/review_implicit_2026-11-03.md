VERDICT: GO — the model, the loss and the sampler have no correctness bug; before the first submit, create the frozen worktree `gpic-wt/dist_step_run` (finding 1).

Review of 1eb4780 (branch dist-step), 2026-11-03. Nothing ran on a GPU. CPU checks ran on vis2. Scripts and outputs: `/scratch/gpfs/GROUP/USER/tmp/review_implicit/` (`check.py`, `check3.py`, `pc.sh`, `fit.yaml`, `pred.yaml`).

## Findings, most severe first

1. **The frozen worktree does not exist, and a missing worktree requeues forever.**
   - `jobs/train_implicit.sh:15`, `jobs/eval_implicit.sh:17` and `jobs/smoke_implicit.sh:14` default to `$WC/project/gpic-wt/dist_step_run`. `git worktree list` does not show it, and `ls` finds no such directory.
   - The smoke script stops with exit 2, which is correct. `train_implicit.sh` does not check: `jobs/train_ladder.sh:31` runs `python $WT/tools/gpu_health.py`. A missing file fails the same way as an unhealthy GPU, so the job exits 98. wq treats 98 as "only waiting": it requeues the job after 30 min and does not count the run (`tools/wq/README.md:35`). The job loops forever, and no error appears.
   - Fix: `git worktree add ../dist_step_run <reviewed sha>` before the first submit. Also add `[ -d "$LADDER_WT/baselines/PixelGen" ] || { echo "no worktree $LADDER_WT"; exit 2; }` to `train_implicit.sh`, after line 15.

2. **`eval_implicit.sh` can pass its 45 min limit.**
   - `jobs/eval_implicit.sh:4` asks for 45 min, and the usage line runs 6 predicts (50, 16, 8, 4, 2 and 1 calls). Each predict and its eval take 50K images.
   - The d640 point smoke took 441 s for predict at 50 steps and 86 s for eval (`runs/gpic-speed/ladder_smoke_1102_1636/exp_ladder_d640/eval_000320/phase_times.json`). One implicit call at m = 1 costs the same as one point step. Each predict also has a fixed cost (start-up, compile, the Qwen text encoder) that I could not measure; I estimate 150–250 s. The total is then about 31–39 min, before the checkpoint check, with every sweep at risk on a slow node.
   - When the job is killed in the middle, the calls that are done keep their `metrics.json`, but the remaining ones are lost, and a fetched guest that runs too long ends its host.
   - Fix: `--time=01:20:00`, or two jobs: `50` alone, and `16 8 4 2 1` together.

3. **The step count in the header is not what the formula gives (+0.24 % FLOPs for the implicit arm).**
   - `jobs/train_implicit.sh:11-12` says 137.7 GFLOP per image and 101216 steps. Recomputed with `tools/ladder_flops.py`'s `fwd` (T = 51.50, E[T²] = 4781 from `docs/ladder_runs.json`), counting 6 blocks once, then 3 blocks plus the head twice, plus the xi layers (0.126 GFLOP over the 3 passes): implicit 137.81, point 104.15, ratio 1.3231. 133600 × 104.15 / 137.81 = **100974 steps**. The header's own numbers give 133600 × 104.2 / 137.7 = 101099, not 101216.
   - The result: the implicit run gets 0.24 % more training FLOPs than the point model at C0/4 (3.571e18 against 3.562e18). The effect is small, but the header's claim is false.
   - Fix: `IMPLICIT_STEPS` = **100992** (a multiple of 32; ms every 25248, last every 3156), and correct the header to 137.8 GFLOP with the xi layers.

4. **All 8 ranks draw the same t, noise, CFG dropout and xi (inherited from the ladder, not new).**
   - `seed_everything: 0` (`lite_ladder.yaml:2`). Lightning seeds every rank with the same seed (`lightning/fabric/utilities/seed.py`; `strategies/ddp.py:200` `reset_seed()`), and nothing in `src/` adds the rank. Every per-step draw has a fixed shape: `t` [32], `noise` [32,3,256,256], the CFG mask [32], and xi [64,256,64] and [64,1,64]. So every rank's CUDA generator stays in step, and rank r's draw j equals rank r′'s draw j.
   - A global batch of 256 images thus has only 32 distinct (t, noise, xi). This does not bias either loss: the two xi of one x_t are still independent. It reduces the variance reduction of the global batch for both arms.
   - At sampling, every prompt starts from the same x_T: `gpic_jsonl_dataset.py:57` seeds with `sample_idx`, which is 0 for every prompt when `num_samples_per_instance: 1`. xi is the same on all 8 ranks at each batch position. So 8 prompts share both x_T and xi. Each image is still a valid draw for its own caption, but draws across prompts are coupled. The point model already has the x_T coupling.
   - Fix: none now. The point d640 run (w975) already runs with this seeding, and a change would break the comparison. Write it in docs/distribution_step.md. For a later version, seed xi per prompt (e.g. a generator seeded from the record key).

5. **The C0/4 comparison has two small asymmetries that come from equal FLOPs, not from code. Report them; do not fix them.**
   - The point model's C0/4 checkpoint is `ms-step=133600` inside a run of 534400 steps (`docs/ladder_runs.json`: ms_every 133600). The implicit arm's C0/4 checkpoint is its last one (101216, or 100992 after finding 3). The LR is warmup 4000, then flat with no cooldown (`lite_ladder.yaml:192-198`), in both arms. So a middle checkpoint and a last checkpoint see the same LR. Good.
   - EMA: decay 0.9999, half-life 6931 steps in both arms. The window is about 10k steps: 7.5 % of the point model's 133600 steps, 10 % of the implicit arm's 101k. If both models still improve at C0/4, the EMA lags by about half-life / run length: 5.2 % of the point run against 6.9 % of the implicit run. This bias is small and works against the implicit arm. The EMA covers the same number of images in both arms, which is the fair way to state it.
   - The implicit arm sees 24 % fewer images (25.9M against 34.2M). This is intended under equal FLOPs.
   - Also, the ladder's point run was launched from commit 6d57702 (`exp_ladder_d640/launches.txt`). The differences from this branch's base are speed changes only: `bucket_cap_mb: 1000`, HostTuning, tcmalloc, the bench timer. They change the bucket order of the gradient all-reduce (bf16 rounding only), not the math.
   - What to do: state in the results that the point model is a middle checkpoint, the implicit model is the last one, and the EMA lag is slightly larger for the implicit model. If the implicit model wins by a small margin, that margin is conservative. A third point, the implicit model at its ms checkpoints (25248 × k), gives the FD–FLOP curve; compare it with the point model's ms checkpoints on the same FLOP axis.

6. **The smoke test asserts nothing on the new code.**
   - `jobs/smoke_implicit.sh` only logs exit codes. A run whose `between` stays at 1e-4 (the samples never separate) or whose loss is NaN passes.
   - Fix: after the first arm, grep the log for `between` and require a finite loss and `between > 1e-3` at step 800. Also require zero recompiles after the warm-up (`TORCH_LOGS=recompiles` is already set).

7. **`timeout` does not reach `fast_eval.py`'s workers (low).**
   - `torch.distributed.run` starts its workers with `start_new_session=True` (`torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py:56`). If eval hangs and `timeout -k 60` sends SIGKILL to torchrun, its workers stay in their own sessions and keep the GPUs. Lightning's ranks use a plain `Popen` (`lightning/fabric/strategies/launchers/subprocess_script.py:144`), so they stay in timeout's group and die with it.
   - Fix: none needed for the smoke test. Its eval arm is last, and the job ends soon after.

## Checked and correct

a. Point model unchanged at xi_dim = 0.
- `check.py` builds the old `JiT_T2I.py` (HEAD~1) and the new one with seed 1. Same state-dict keys and values. Same CPU RNG state after init, so no extra RNG draw at init. Same forward output, bit for bit, and the same RNG state after the forward.
- `xi_*` modules exist only when `xi_dim` is set (`JiT_T2I.py:291`). `initialize_weights` touches them only then (`:346`).
- Training, sampling and checkpoint keys are therefore the same as at the merge base. The compile cache keys change, because the code object and the signature changed. That costs one recompile and no numerical change. The ladder's production runs use another worktree anyway.

b. torch.compile, DDP, EMA.
- `m` is a Python int keyword argument. Dynamo specializes on it. The denoiser sees only m = 2 (the trainer and CompileWarmup call the same `diffusion_trainer`), and the EMA net sees only m = 1 (sampling). On CPU with inductor, changing T and m gave 4 graphs and no error.
- `torch.randn` inside the compiled graph gives new noise on every call: `check3.py` "fresh noise per call: True". The backward uses the forward's noise, not a second draw: the weight gradient equals the hand gradient computed from the forward's noise, with and without returning the noise. The reason: random ops are never recomputed (`torch/_functorch/partitioners.py:1065`, `random_ops` at `:1513`; `functionalize_rng_ops = False`). On CUDA, inductor seeds its Philox from the same generator (`torch/_inductor/inductor_prims.py:64`).
- dtype at the fork under bf16 autocast: `x` is bf16 after `x_embedder` (`x += pos_embed` keeps bf16), `randn(..., dtype=x.dtype)` is bf16, and the Linear runs in bf16. The compiled CPU output is bf16 with shape [B·m, C, H, W].
- DDP wraps the LightningModule. The denoiser is called inside `training_step`, so `m` never passes through DDP's forward. The xi layers are in the autograd graph at every step, because training always forks.
- `xi_global` gets exactly zero gradient at step 0, because every block's adaLN output layer is zero at init (`JiT_T2I.py:335-337`). It is still in the graph, so `find_unused_parameters: false` is fine. It moves once adaLN moves. This is expected.
- The EMA (`simple_ema.py:76-77`) and `copy_params` (`copy.py`) zip `parameters()` of a deepcopy, so the xi parameters line up.

c. Energy score.
- The two samples of one x_t get independent xi: separate rows of one `randn` after `repeat_interleave`.
- At a fixed t, w and the RMS (1/√n) are positive scalars on the norm. They multiply the score by a constant, so propriety holds.
- The 1e-8 floor: (r² + ε)^{β/2} is a Bernstein function of r², so the kernel score stays strictly proper. At init the two samples are equal (zero output layer). The gradient of √(r² + ε) at r = 0 is 0, not NaN.
- With m = 2 the "between" term is the single pair, which is unbiased. `x` and `noise` are fp32. `g` is bf16, then `.float()`. The point model has the same bf16 output.
- CFG dropout: `preproprocess` changes `condition` and `metadata["y_mask"]` per sample. The fork applies `repeat_interleave` to `y`, `condition` and `full_mask` together, so each copy keeps its own caption and mask.
- `loss_per_sample` is [B] and is popped as before. `to_data` and `between` are detached tensors in `log_dict`.

d. Sampler.
- With timeshift 1.0 the times are linspace(0, 1 − 1/N, N), then 1.0: N equal steps. The last step gives x + (1/N)(x̂ − x)/(1/N) = x̂, and `test_implicit.py` check 6 passes for 1, 2, 8 and 50 steps.
- N = 1: `last_step` = 1.0, so the times are [0, 1] and the output is x̂.
- `1 − t_cur` ≥ 1/N, so there is no division by zero.
- The predict wrapper passes `y_mask` directly, because CFG is off and `x.shape[0] == _m.shape[0]`. With m = 1 the fork is a copy plus xi. `BaseSampler.forward` returns `x_trajs[-1]`.
- `guidance: 1.0` is asserted. `_sample_and_log` (CFG > 1) is off in training (`sample_every: 0`), and training keeps `EulerSamplerJiT`.

e. Jobs.
- jsonargparse 4.41.0, `--print_config` with the exact command lines. In `fit`: hidden_size 640, num_groups 10, 9 + 2 blocks are kept, and the later `--config` adds xi_dim 64 and xi_block 6. The trainer class becomes `ImplicitJiTTrainer` with every init_arg (the overlay lists them all), samples 2, beta 1.0. In `predict`: `ImplicitSampler`, `num_steps` 8 from the flag, guidance 1.0, xi set.
- The run directory: main.py makes `<root>/exp_<tags.exp>` = `exp_ladder_implicit_d640`, the same as `train_ladder.sh`'s `RUN`. So resume finds `last.ckpt` and eval finds `ms-*`. The W&B name is `ladder_implicit_d640`, the id `ladder-implicit_d640`.
- 101216 and 3200 are multiples of 32.
- The scripts are self-contained, except the documented smoke-only variables. `train_implicit.sh`: `fetchable-partial`, 2.5 h. My estimate is about 1.2–1.6 h at about 1.32 × the 32.5 ms point step; it resumes from last.ckpt.
- `set -eo pipefail` with the `python -` heredoc stops the script on a failed check. The check rejects a checkpoint that is not implicit.
- The smoke test: the later `--trainer.max_steps 800` wins, the ms checkpoint lands at 800 (3200/4), the timeouts sum to 32 min plus 3 min of `-k`, within 40 min, and a failed arm does not stop the next arm.
- `timeout` kills Lightning's ranks with its process group.

## Not checked

- Anything on CUDA: the step time, memory at m = 2, DDPOptimizer's graph splits around the fork, and whether inductor's CUDA RNG gives new xi on every call in a CUDA graph (it is not used; `GPIC_PROF_DEN_COMPILE_MODE` is unset).
- The fixed cost of one predict (finding 2).
- Whether the energy score trains well at d640: the toy (`tools/dist_step/toy_implicit.py`) is the only evidence, and it shows the implicit model losing at 50 calls.
- The design choices themselves: xi_block 6, xi_dim 64, w = 1/(1−t).

## Fixes (2026-11-03)

- 1: the frozen worktree gpic-wt/dist_step_run is created at the fixed commit; train_implicit.sh exits 2 if it is missing.
- 2: eval_implicit.sh asks 1:20:00.
- 3: 100992 steps; the header states the recomputed numbers.
- 4, 5: stated in docs/distribution_step.md. The same-seed fact is reported to the owner: it is in the official recipe and in the ladder.
- 6: the smoke job checks the score, `between` and recompiles (GPIC_IMPLICIT_PRINT, unset in production).
- 7: no change.
