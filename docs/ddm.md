# Distributional denoiser on the final ladder (ddm, P1)

Question: at the same training FLOPs, does a denoiser that draws samples of p(x | x_t) beat the point model at 4 to 8 network calls, in pixel space?

Owner decisions (2026-11-22, via the controller): 02:1x 「genmodel可以开始探索能生成分布的生成模型了」; popup option A: P1 (this) and P2 (few-call FD of the
point and K = 8 mixture checkpoints, w3748); 03:xx option A: P1 at d10 only first; d12 only if d10 beats the point model at 4 to 8 calls; if d10 loses
too, stop and propose a feature-space or latent variant.

Prior result in this repository: the implicit sampler of 2026-11-05 (w1056, d640, old recipe with EMA; additive noise in the last 3 of 9 blocks plus a
global noise in the blocks' conditioning, 2 samples, whole-image energy score, beta 1, lambda 1) lost at few calls: FD 2401 / 1907 at 4 / 8 calls against
the point model's 1385 at 8.

## Recipe

iDDM (Martorella et al., arXiv 2609.37147, ImageNet-256 latents), config F of their Table 1, taken as published except where pixel space forces a choice:

| | iDDM-B/2 | here (d10) |
|---|---|---|
| samples per image, m | 4 | 4 |
| per-sample blocks | last 2 of 12 | last 2 of 10 (`xi_block = nb - 2`) |
| noise | 256 channels per token, concatenated, fixed inner width | same (`xi_mode cat`, `xi_dim 256`) |
| noise gate | xi (1 + W c), W zero at init | same, c = the blocks' conditioning input |
| score | generalized energy score per token (local kernel) | same, per 16 x 16 x 3 token |
| norm in the score | not stated | root mean square per token (below) |
| lambda(t), beta(t) | linear: 1 - t; 2 - 1.9 (1 - t) | same (t = 1 clean in both) |
| time sampler | logit-normal (-0.8, 0.8) | the ladder's, the same |
| prediction | velocity | x, scored in v space (JiT), as the point model |
| sampler | Euler on the drawn velocity, fresh noise per call | same (`ImplicitSampler`) |
| training | AdamW, EMA, CFG | the final ladder: Muon + AdamW, no EMA, WSD 50% cooldown, no CFG |

The one choice pixel space forces: a latent token holds 16 (B/2) or 128 (T2I) values, a pixel token 768. With beta(t) moving from 0.1 to 2, a plain L2
norm would scale the score at each t by 768^(beta(t) / 2), a factor that changes about 550-fold across t and would act as a time weighting iDDM
never chose (they keep w(t) = 1 and let p(t) weight the times). The root mean square makes the score's scale independent of the token size.

FLOPs: the last 2 blocks run 4 times at width d + 256 and the final layer 4 times: r = 0.8237 at d10 (tools/ddm_ladder/steps.py; checked against
FlopCounterMode in tools/ddm_ladder/test_ddm.py), so the run takes 11648 steps against the point run's 21248, at 3.6e18 FLOPs.

## Measurements

- held-out (tools/heldout_loss.py --es, 4 draws per image and level): the squared error of the draws' mean with their variance removed, an unbiased
  estimate of the squared error of the model's own mean, the quantity of the point model's held-out loss; and the energy score per token (beta 1,
  lambda 1), strictly proper, for both models (the point model's is that of the distribution with all its mass on its prediction).
- FD (DINOv2 and Inception, 50K test captions, official noise, no CFG) at 4, 8, 16 and 50 calls (jobs/evals_ddm_ladder.sh); the point model's 4 and
  8 calls come from P2 (w3748), 16 and 50 exist.

## Code

Branch ddm (from gmflow-ladder 14fa39c), frozen worktree gpic-wt/ddm_run. JiT_T2I `xi_mode cat`; ImplicitJiTTrainer `kernel_patch`, `schedule`
(defaults keep the old trainer); configs_h100/ddm_train.yaml, ddm_sample.yaml; jobs/train_ddm_ladder.sh, evals_ddm_ladder.sh, heldout_ddm_ladder.sh,
smoke_ddm_ladder.sh; eval_gmflow.sh sampler `ddm`. Tests: tools/ddm_ladder/test_ddm.py (the point model and the old implicit sampler unchanged bit for
bit; the widened blocks reduce to the point model when their extra weights are zero; the score against loops; Muon parts; FLOPs).
Ledger: project/gpic/logs/dist_ddm.yaml. Page: exps/gpic-speed.html fig-dist-ddm.
