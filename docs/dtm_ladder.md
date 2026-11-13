# The DTM flow head on the final ladder (branch `dtm-ladder`)

Owner, 2026-11-13: 「都可以试试」 after reading Transition Matching (Shaul, Singer, Gat, Lipman 2025, arXiv 2506.23589); popup 00:5x: three trainings
(d10, d12, d14, FLOP-matched to the final ladder's point runs), evaluation at 50 and 16 calls for the flow head and the point model, a separate
figure on the experiment page.

## What it is

DTM samples each step from a learned transition distribution. In our notation (t = 0 noise, t = 1 image, x_t = t x + (1 - t) e):

1. The backbone (the ladder's JiT, unchanged) reads x_t and gives one token per patch.
2. A small flow head (TokenFlowHead: width 1024, 3 blocks, 19.4 to 20.0 M parameters; branch ar's module) draws the patch's 768 values from
   p(patch | x_t, y) given that token. Patches are drawn independently of each other; the values inside a patch are drawn jointly.
3. The state takes the straight step along the draw: x' = x + (sigma - sigma') (x_hat - x) / sigma, sigma = 1 - t, with no fresh noise. The last
   step returns the draw. This is the paper's X_{t+1} = X_t + Y / T with Y = x - e, and our gm_draw with the flow head in place of the per-pixel
   mixture.

## Code

| Part | Where |
|---|---|
| Model | `JiT_T2I(dtm_width=1024, dtm_depth=3, dtm_steps=32, dtm_mean_draws=4)`: FinalLayer's linear is dropped; `dtm_head` reads `norm_final`'s tokens. Default output = the head's mean (x prediction at head time 0, mean over 4 fixed head noises); `dtm_mode="draw"` = one draw (32 Euler steps of the head's flow); `dtm_q`, `dtm_tau` = the head's x_hat for a given head state (training). |
| Loss | `training_dtm.DTMTrainer`: the point model's t and x_t (same distribution, same RNG order), then per patch a head time tau ~ logit-normal(-0.8, 1.6) and noise; v-space L2 with t_eps 0.05 (branch ar's head recipe). |
| Sampler | `sampling_dtm.DTMDrawSampler` (`dtm_draw`); the control is EulerSamplerJiT on the head's mean (`euler_mean`). No CFG. |
| Config | `configs_h100/dtm_train.yaml` (overlay after `muon_ladder.yaml`), `configs_h100/dtm_draw_sample.yaml`. |
| Jobs | `jobs/train_dtm_ladder.sh`, `jobs/evals_dtm_ladder.sh` (50 and 16 calls), `jobs/heldout_dtm_ladder.sh`, `jobs/evals_point_calls.sh` (the point runs at 16 calls with the ladder's own evaluation code), `jobs/smoke_dtm_ladder.sh`. |

With `dtm_width = 0` the model is the point model: every backbone tensor of a DTM model equals the point model's init at the same seed, and the
CPU RNG ends in the same state (the head is built after `initialize_weights` inside `fork_rng`).

## FLOP matching

r = (head forward FLOPs - the point model's final linear) / backbone forward FLOPs per image; steps = point steps / (1 + r), a multiple of 32
(`tools/dtm_ladder/steps.py`). The ladder's FLOPs count the networks only; the head runs once per patch per training image.

| Run | Shape | r | Steps | Point run |
|---|---|---|---|---|
| `dtm-d10-0p25c0` | 768 x (10+2) | 0.176 | 18048 | `xl-d10-0p25c0` (21248) |
| `dtm-d12-1c0` | 1024 x (12+3) | 0.084 | 37088 | `xl-d12-1c0` (40192) |
| `dtm-d14-4c0` | 1280 x (14+3) | 0.047 | 85568 | `xl-d14-4c0` (89600) |

At evaluation one draw costs 32 head passes per call: 5.8, 2.8 and 1.6 times the backbone at d10, d12 and d14.

## Held-out loss

The head is scored by its mean, the same quantity as the point model's loss. `tools/dtm_ladder/mean_check.py` compares that mean with an
unbiased U-statistic over 4 head draws (the smoke runs it).

## Tests (CPU, a vis node)

`tools/dtm_ladder/test_dtm.py` (D1-D6: init equal to the point model's, the three output modes by hand, the trainer's draws and loss by hand,
the sampler's steps, training with no EMA and an exact resume, Muon marks), `tools/test_heldout_loss.py` (H9: a DTM run scored end to end),
`tools/gmflow_ladder/test_port.py --ref <5ac69ab>` (the point model bit for bit), `tools/gmflow/test_gmflow.py` (T1-T33). All pass.

## Differences from the paper

1. Pixel space: a patch has 768 values, the paper's latent token 16 (2 x 2 x 4 of SDXL-VAE).
2. The ladder's t distribution (logit-normal), not the paper's uniform t over 32 discrete steps.
3. No CFG (the paper uses 6.5 on the head's velocity and reports no number at CFG 1).
4. FLOP-matched training, not equal steps.
