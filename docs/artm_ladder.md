# ARTM on the final ladder (branch `artm-ladder`)

Owner, 2026-11-13: after Transition Matching (Shaul, Singer, Gat, Lipman 2025, arXiv 2506.23589) 「都可以试试」; popup: ARTM-3 (not FHTM), at all three
budgets (「直接三档」). Held-out loss is not comparable for this model (no mean per noise level); it is judged by FD.

## What it is

Generation takes T = 3 transitions X_0 -> X_1 -> X_2 -> X_3 = image, X_0 = the initial noise (the official noise at evaluation). Training uses the
independent linear process X_k = (1 - k/3) e_k + (k/3) x, with a new noise e_k for every k (the paper finds the shared-noise process far worse for
autoregressive kernels). Each transition is autoregressive over the 256 patches, as an LLM: the model gives the distribution of patch i of X_{k+1}
given all of X_k, the patches of X_{k+1} before i and the caption, and the flow head draws it.

## Code

| Part | Where |
|---|---|
| Model | `ARTM_JiT.ARTMJiT(art_T=3, art_head_width=1024, art_head_depth=3, art_head_steps=32)`: the ladder's JiT_T2I blocks on 512 image tokens (X_k's patches, full attention among themselves; X_{k+1}'s patches shifted by one after a learned BOS, causal), both halves with their patches' 2D positions and a learned segment vector; adaLN on the time embedding of k/3; FinalLayer's linear dropped; `art_head` (TokenFlowHead) on the target tokens. |
| Loss | `training_artm.ARTMTrainer`: k uniform in {0, 1, 2}, two new noises, X_k and X_{k+1}; per target patch a head time tau ~ logit-normal(-0.8, 1.6); v-space L2 with t_eps 0.05 (the DTM head's recipe). |
| Sampling | `ARTMJiT.artm_sample` (called by `predict_step` for `is_artm` models): per transition one pass over the context with its keys and values kept per block, then 256 one-token steps; the head draws each patch (32 Euler steps of its flow). |
| Config | `configs_h100/artm_train.yaml` (overlay after `muon_ladder.yaml`), `configs_h100/artm_sample.yaml`. |
| Jobs | `jobs/train_artm_ladder.sh` (SDPA attention, micro-batch half the point model's, twice the accumulation), `jobs/evals_artm_ladder.sh` (sampler `artm`, "calls" = 3), `jobs/smoke_artm_ladder.sh`. |

With art_T unused the point model is unchanged; ARTMJiT's backbone starts at the point model's init at the same seed (the new parameters are built in `fork_rng`).

## FLOP matching

**Superseded (owner, 2026-11-13 07:4x: 「不需要对齐算力 对齐所有训练细节就行」).** The runs train with the point runs' training details: steps 21248 / 40192 / 89600 at d10 / d12 / d14, the same Muon and AdamW rates, Muon decay 0.0173 / 0.01829 / 0.01641, batch 1024, 50% cooldown, GPIC Full; the queued jobs' arguments were changed before any started (tmp/ladder_prof/dist_phase2/align_steps_1113.py). The FLOP-matched rows below are kept for the record.

ARTM's forward FLOPs per training image: 512 image tokens through the blocks, attention on the keys the mask allows, the caption as the point model's,
the head on 256 patches (`tools/artm_ladder/steps.py`, the ladder's own per-part formula). Steps = point steps x point FLOPs / ARTM FLOPs.

| Run | Shape | FLOPs per image vs the point model | Steps | Point run |
|---|---|---|---|---|
| `artm-d10-0p25c0` | 768 x (10+2) | 2.13 x | 9984 | `xl-d10-0p25c0` (21248) |
| `artm-d12-1c0` | 1024 x (12+3) | 2.03 x | 19840 | `xl-d12-1c0` (40192) |
| `artm-d14-4c0` | 1280 x (14+3) | 1.99 x | 44960 | `xl-d14-4c0` (89600) |

So at equal compute ARTM sees about half the images the point model sees. The paper compared at equal training steps.

## Tests (CPU, a vis node)

`tools/artm_ladder/test_artm.py` (A1-A7): the init equals the point model's; the mask (a target position reads only earlier target patches, the
context and the real caption tokens); the KV-cached sampling equals the full masked forward to 2e-6 at every transition and position, with and
without a caption mask; the process and the loss by hand; sampling; training with no EMA and an exact resume; Muon marks. All pass. The GPU smoke
repeats the cache check in bf16 with and without the fused norm (`tools/artm_ladder/kv_check.py`).

## Differences from the paper

Pixel patches of 768 values (the paper: latent tokens of 16); a DiT-style backbone with joint image-text attention (the paper's DiT has cross
attention); no CFG (the paper: 6.5); FLOP-matched training (the paper: equal steps).
