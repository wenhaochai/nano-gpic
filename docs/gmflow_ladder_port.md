# The per-pixel mixture on the final ladder (branch `gmflow-ladder`)

The port puts the per-pixel Gaussian mixture (K = 8, with K = 1 as the control) on the final ladder's trainer and leaves the point model's
path bit for bit as it was, so a mixture run and its point run differ only in the output head and the loss.

Phase 2 of the project asks one question (owner, 2026-11-12): does sampling by the distribution's probabilities at every step lose to the
point model's deterministic sampling? The pass line is "not worse", at 50 calls, on the final ladder's budgets.

## 1. Code

| Commit | Content |
|---|---|
| `903adfe` | merge of `ladder-full-gmflow` (`dceebd6`, the mixture) into `launch-retry-cap` (`5ac69ab`, the final ladder's trainer) |
| `6e392c8` | merge of `eval-official-run11` (`abdb631`, the ladder's evaluation code) |
| `da07b46`, `9e6dfe8` | the ladder's held-out tool, then the mixture in it (scored by its mean; `--nll`) |
| `e185058`, `99d6e3b` | the ladder's launcher, then the mixture launcher, the step calculator and the port tests |
| `c01235a`, `14fa39c` | evaluation, held-out and smoke jobs |

Frozen worktree for jobs: `gpic-wt/gmflow_ladder_run`.

## 2. What differs from the point model

| Part | Point model | Mixture (gm_k = K) |
|---|---|---|
| Final linear | 768 outputs per token (3 x 16 x 16) | K x 1024 outputs per token: K means (3 channels) and K logits per pixel |
| Spread | none | one log s per image, from a small MLP on the time embedding plus the pooled caption (`gm_logstd`) |
| Loss | MSE on the velocity (`JiTTrainer`) | GMFlow's transition NLL, lambda 0.5 (`GMFlowTrainer`, `configs_h100/gmflow_train.yaml`) |

Everything else is the ladder's recipe, argument for argument: Muon on the block matrices and AdamW on the rest (the head is in the AdamW
groups, as the point model's final layer is), no EMA, warmup 1%, linear cooldown over the last 50% of the steps to 3e-5, batch 1024, the same
time sampling (lognormal, mean -0.8, std 0.8), no caption dropout, bf16, gradient clip 1.0, GPIC Full, FSDP2 at d14.

## 3. FLOP matching

**Superseded (owner, 2026-11-13 07:4x: 「不需要对齐算力 对齐所有训练细节就行」).** The runs train with the point runs' training details: steps 21248 / 40192 / 89600 at d10 / d12 / d14, the same rates, Muon decay 0.0173 / 0.01829 / 0.01641, batch 1024, 50% cooldown, GPIC Full; the queued jobs' arguments were changed before any started (tmp/ladder_prof/dist_phase2/align_steps_1113.py). The FLOP-matched rows below are kept for the record.

r = the head's added forward FLOPs / the point model's forward FLOPs per image. Steps = point steps / (1 + r), rounded down to a multiple of
32. Muon decay = 5 / (Muon LR x steps). `tools/gmflow_ladder/steps.py` prints the rows.

| Run | Shape | r | Steps | Point run (steps) | FLOPs |
|---|---|---|---|---|---|
| `gm8-d10-0p25c0` | 768 x (10+2) | 0.0535 | 20160 | `xl-d10-0p25c0` (21248) | 3.6e18 |
| `gm1-d10-0p25c0` | 768 x (10+2) | 0.0019 | 21184 | | 3.6e18 |
| `gm8-d12-1c0` | 1024 x (12+3) | 0.0337 | 38880 | `xl-d12-1c0` (40192) | 1.4e19 |
| `gm1-d12-1c0` | 1024 x (12+3) | 0.0012 | 40128 | | 1.4e19 |
| `gm8-d14-4c0` | 1280 x (14+3) | 0.0235 | 87520 | `xl-d14-4c0` (89600) | 5.7e19 |
| `gm1-d14-4c0` | 1280 x (14+3) | 0.0008 | 89504 | | 5.7e19 |

## 4. Jobs

| Script | Does |
|---|---|
| `jobs/train_gmflow_ladder.sh <row>` | one training; K from the tag; run in `runs/gpic-speed/gm_ladder/<tag>`; resumable |
| `jobs/evals_gm_ladder.sh <tag>:<d>:<nb>:<nt>:<step> ...` | raw view, then per run `gm_draw`, `gm_sde`, `euler_mean` at 50 calls on 50K captions (official noise, no CFG), one chain; DINOv2 and Inception metrics |
| `jobs/heldout_gm_ladder.sh` | held-out loss of the mixture mean and the NLL per level for every finished run; 4 GPUs; records in `tmp/heldout_loss/heldout_loss_gm.jsonl` |
| `jobs/smoke_gm_ladder.sh` | the GPU smoke test (section 6) |

Samplers: `gm_draw` draws one component per pixel by the mixture's probabilities at every call, then takes the straight step (the main
result). `gm_sde` is the same draw plus fresh noise (reference). `euler_mean` is Euler on the mixture mean (control). The point model's
numbers are the ladder's own records (Euler, 50 calls, the same noise and captions).

## 5. CPU tests (vis node)

`bash tools/gmflow_ladder/vis.sh <test file>`:

- `tools/gmflow_ladder/test_port.py --ref <export of 5ac69ab>`: all pass. Groups: every parameter in exactly one group, the head in AdamW,
  the same Muon marks as the point model. Training: finite losses with no EMA and the cooldown, the EMA slot equals the raw weights in every
  checkpoint. Resume: the same weights and schedule after a stop. The point model: losses, groups and final weights equal `5ac69ab` bit for
  bit. The head's added FLOPs equal the formula of `steps.py`.
- `tools/gmflow/test_gmflow.py`: T1 to T33 pass (the old mixture suite).
- `tools/test_heldout_loss.py`: H1 to H7 pass. H7: the mixture's loss with and without `--nll` agree, and the NLL per dimension equals a
  brute force to 2e-7 (K = 8 and K = 1).

## 6. GPU smoke test

`jobs/smoke_gm_ladder.sh` (wq w2619): 320-step trainings of gm8 at d10 with a stop and a resume, gm8 at d14 (FSDP2) and gm1 at d10; the
evaluation and held-out jobs on their checkpoints; the ladder's d10 point run evaluated again with this code (its FD must match the ladder's
record within 1%). Result: see `tmp/ladder_prof/dist_phase2/STATUS.md`.

## 7. Known limits

- A per-pixel draw ignores the correlation between pixels. The old runs measured its cost only at small compute and with the old recipe;
  the ladder runs measure it again at 50 calls.
- The mixture trains with one more loss term per pixel and component. The FLOP count follows the ladder's convention (network FLOPs), so
  this time is not in the budget; the smoke test reports the step time against the point run's.
