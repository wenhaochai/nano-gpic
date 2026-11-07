VERDICT: FINDINGS (12)

Strict audit of the results reported on 2026-11-04. Read-only, except this file and helper scripts in `tmp/audit_1104/` (anova.py, arm_precise.py, toy_chain.py). All analysis ran on vis2 with no GPU.
Abbreviations: page = `tmp/gpic_site_wt/exps/gpic-speed.html`, figs = `tmp/gpic_site_wt/assets/data/gpic-speed-figures.js`, record = `.../gpic-speed-record.js`, doc = `docs/distribution_step.md` (this worktree), tc = `project/gpic-wt/ladder/docs/text_cache.md`.

## Findings, most severe first

### 1. "The variety among the samples comes only from the captions" and "77% of the sample variance comes from the noise" mislead together (high)
- Where: page:174 (EN), page:175 (ZH "样本之间的差异只来自 caption"), page:176/177 ("77% of the sample variance comes from the noise"). Doc 5.3 item 1 (doc:295, "多样性只来自 caption。协方差少了噪声贡献的部分").
- Source: `noise_eval_2bd71b7/*/features.npy`. I ran a two-way decomposition (caption × tensor) on the three shared tensors (official, fixed1, fixed2; 50,000 captions, 1024 features):
  - caption main effect 317 (24%), noise main effect 78 (6%), interaction 925 (70%). The sum is 1320, and the per-prompt trace is 1318.
  - With the caption fixed, the variance over tensors is 1003 (76%). The script's 77% (1013 / 1318) is the same quantity. The number reproduces.
  - Trace of the covariance: per-prompt 1318 (prompt_b 1318); fixed1 1264, fixed2 1278, official 1184. One shared tensor removes 3%, 4% and 10% of the trace. It does not remove 77%.
  - The shared tensor changes the shape of the covariance. Participation ratio: fixed1 52, per-prompt 101. Top eigenvalue: 118 against 73. The top 10 directions hold 34% against 25%.
- Why it matters: a reader joins the two sentences and concludes that sharing removes 77% of the variance and that this explains the 28% FD gap. The data say that a shared tensor keeps 90–96% of the variance, because the noise acts differently for each caption. The FD gap comes from the mean (fd_mu 561 to 428) and from the shape of the covariance (fd_sigma 943 to 656), not from a smaller trace. The doc's own reading at doc:354 ("大部分是噪声和 caption 的交互") is correct. The page dropped it.
- Correct wording, page:174: "So all 50,000 prompts start from one noise tensor, and the noise no longer varies from prompt to prompt." Page:176: "With the caption fixed, a change of the noise moves the DINOv2 features by 77% of their total variance. One shared tensor still leaves 90 to 96% of the variance, because the noise acts differently for each caption, but it concentrates the variance in fewer directions." Doc:295 needs the same fix.

### 2. The 2–4% "between two runs of one recipe" comes from two different recipes (medium-high)
- Where: figure fig-code, data-more-en/zh (page:170/171, "two runs of one recipe with different seeds differ by 2 to 4%"). Page:162/163 ("the 2 to 4% between two runs of one recipe"). Doc 5.3 rule 1 (doc:304, "seed 测试显示同配方两次运行差 2–4%").
- Source: `ladder_seed_same` against `ladder_seed_rank` (w1053, w1054): 2099.06 against 2144.16 (+2.1%) and 1503.63 against 1451.32 (−3.5%). One arm uses the official seeding, and the other uses one seed per GPU. Page:137 itself calls these two seedings.
- Why it matters: this one pair is the only noise estimate behind the 5% rule and behind every "within" claim. It is one pair, and the two runs differ in their seeding scheme, not only in the seed.
- Correct wording: "two runs that differ only in how the GPUs are seeded differ by 2.1% and 3.5%; one pair gives no noise estimate."

### 3. "So the gain appears only at low compute" states a single-run gap as an effect (medium)
- Where: page:162 ("So the gain appears only at low compute"), page:163 ("所以收益只出现在低算力处"). Doc 5.6 item 6 (doc:413, "方案 A 的收益在低算力处").
- Source: one run per arm. Per-prompt noise: 64 nats −8.87% / −1.69%, 256 nats −10.29% / +1.86% (my computation; the point model at C0/8 is the mean of two draws, 1083.93). The rule needed both points and a second seed, because one pair cannot estimate noise (doc:305).
- Why it matters: the paragraph's next sentence says that one pair gives no noise estimate. Then "So the gain appears" contradicts it. At C0/16 the gap is 2.5–5 times the one seed-pair difference, which is suggestive but not established.
- Correct wording: "In these single runs the code is 9% and 10% lower at a sixteenth of the compute and within 2% at an eighth."

### 4. "The network hardly uses the code to choose the image" is stronger than the data (medium)
- Where: page:164/165. Doc 5.5 (doc:375, "enc_match 很低（随机是 0.004），所以主网络基本没有用 z 决定画哪张图").
- Source: `code_overfit2_ddcae31`, step 24000. enc_match: 16 nats 0.011 and 0.021 (3 and 5 times chance), 64 nats 0.117 (30 times chance). Chance is 1/256 = 0.0039. These numbers match the text.
- Problems:
  - At 64 nats the code steers 12% of the samples, 30 times chance. "Hardly uses" does not fit this rate.
  - The test does not isolate the network. `sample_enc` (overfit_code.py:97–99) gives the network a fresh noisy z = a·u_hat + sqrt(1−a²)·ξ. The channel power a² is 0.39 at 16 nats and 0.86 at 64 nats. A low enc_match can mean either that the network ignores z, or that the noisy z does not identify the image. No test decodes the source image from z, for example with the nearest centre. Without it the cause is open.
  - enc_match is argmin over the training images, with no distance threshold (overfit_code.py:160–161). "Land on that image" suggests the near threshold of 0.1, which enc_match does not apply.
  - near against enc_near (0.695/0.703, 0.582/0.559, 0.667/0.679) supports "the flow is not the limit" on this set. That half of the sentence holds.
- Correct wording: "With a code from the encoder, the nearest training image of a sample is the code's own image for 1 to 2% of samples at 16 nats and 12% at 64 nats, where chance is 0.4%. The code steers the image only weakly. We did not test whether the noisy code still identifies its image."

### 5. The CPU budget: w1117 does not confirm the 26.7 ms limit, and the 8.97 ms rests on one control (medium)
- Where: tc:130 ("This confirms the last section"). tc:115 ("The 96 CPUs set the limit at d640 … 26.7 ms"; "Only less worker CPU per image can move the mean"). Page:139/140 ("The 96 CPUs set a limit near 26.7 ms per step").
- Source: tmp/ladder_prof/host_prof.
  - 8.97 ms comes from the first control of w1094 and its thread_wait.txt (78.4 worker CPUs at 29.3 ms). The arithmetic is right: 8737 images/s, (96 − 10) / 8.97 ms gives 26.70 ms.
  - The same arm's cpu_wait.txt gives 74.0 CPUs (8.47 ms). Control 2 gives 77.5 (thread_wait, 8.90 ms) and 70.7 (cpu_wait, 8.12 ms). The bound is therefore 24.2 to 26.7 ms, depending on which control and which tool.
  - w1117, crop arm: thread_wait gives 69.3 worker CPUs at 26.73 ms, which is 7.23 ms per image. The same-run controls give 8.91 ms. That is 19% less, and by the doc's own formula the new bound is about 21.5 ms. The crop arm runs at 26.7 ms, 5 ms above its new bound. Crop with pinning runs at 26.4 ms. So after the crop, the CPU budget no longer sets the mean.
  - That the crop's mean (26.7) equals the old bound (26.7) is a coincidence. The bound was computed for the old CPU cost per image.
- Why it matters: the text presents w1117 as confirming a quantitative limit. w1117 confirms the direction only: with 19% less worker CPU per image, the mean falls 9.3%, and the pin stalls vanish (crop with pinning has 0 steps over 0.3 s, against 9 in w1094). That supports "pinning fails because the workers lack CPU" (page:139). It does not support "the limit is 26.7 ms", and it does not support "only less CPU can move the mean". The prediction at tc:115 ("20% less would put the limit at 21.4 ms") was met as a bound, but the mean did not follow.
- Correct wording, tc:130: "This agrees with the direction of the last section: 19% less worker CPU per image (8.91 to 7.23 ms) lowers the mean 9.3% and removes the pin stalls. The new mean, 26.7 ms, is above the new CPU bound of about 21.5 ms, so the CPU budget no longer sets the mean." Page:139: "The 96 CPUs bound the step at about 24 to 27 ms at this width."

### 6. One colour has different meanings in fig-code and fig-noise (medium; owner rule)
- Where: figs:187 and figs:207.
- Source: in fig-code, `--ch-s1` is 64 nats, `--ch-s2` is 256 nats, and REF (dashed) is the point model. In fig-noise, `--ch-s1` is "one noise per prompt" and `--ch-s2` is "one tensor for all prompts", and these bars are the point model. The same numbers (1529.61, 1083.9x, 2099.06, 1503.64) appear in REF in fig-code and in s1/s2 in fig-noise. Elsewhere `--ch-s1` is "our run" (fig-lite-ref), "matrix multiplies" (fig-speed), "C0" (fig-ladder) and "posterior sampler" (fig-implicit). `--ch-s2` is "exposed communication" and "3C0/4". The header at figs:3 promises one colour per setup only "across panels".
- Why it matters: a reader of the two neighbouring figures reads blue as 64 nats in one and as per-prompt noise in the other, for the same point model.
- Fix: draw fig-noise's bars in the point model's colour (REF), and tell shared from per-prompt by two tones of it or by the row names. Keep s1 and s2 for the code rates.

### 7. "The model underestimates its error" is only partly supported (medium-low; doc only, marked for discussion)
- Where: doc:364 ("所以这个模型低估了自己的误差"). Doc:431 calls the total error 0.185 "模型误差".
- Source: `posterior_mc_1104_1357/shard*.json` (4 × 32 images). The table at doc:360–362 reproduces exactly: err 0.422 … 0.060, spread 0.295 … 0.015, gap 0.083 … 0.046.
- What the data allow:
  - err²/spread² is 2.05, 1.81, 1.86, 1.97, 2.41, 3.63 and 15.5 at t = 0.02 … 0.9.
  - The grid: I ran the same chain (same 50-step grid, same update) with an exact posterior mean on two toy data sets (`tmp/audit_1104/toy_chain.py`). Gaussian data: the spread is 5% low at t ≤ 0.5 (ratio about 1.1), and the ratio is 2.0 at t = 0.9. 256 points in 768 dimensions: the ratio is 1.00–1.04 at t ≤ 0.2. So the grid explains little of the real ratio of about 2 at t ≤ 0.5, and less than 2 of the 15.5 at t = 0.9.
  - The chain's mean misses x_hat by 0.08–0.09, while 32 samples alone would give 0.003–0.05 (spread / sqrt(32)). The model's later steps do not agree with its x_hat. The doc says this.
  - A point model makes no estimate of its own error. The chains' spread comes from the sampler's injected noise, propagated through later x_hat. For a near-exact chain, err² > spread² holds for any model with nonzero error: err² is the posterior variance plus the squared model error. The result therefore measures the size of the model's error (about 0.13 rms at t = 0.3, from err² − spread²). It is not a separate trait of "underestimating".
  - The 128 images come from training shard 0, which the model saw about 1.9 times (66816 × 256 / (801 × 11042)). On held-out images the error is likely larger. Untested.
- Correct wording: "On 128 training images the real error exceeds the chains' spread at every t (err²/spread² 1.8–2.4 for t ≤ 0.5, 15 at t = 0.9). With an exact mean, the same chain on toy data gives 1.0–1.1 at t ≤ 0.5, so most of the gap is model error, not the grid. Held-out images are not tested." At doc:431, write "总误差" instead of "模型误差".

### 8. "The flow costs under 0.1% of a forward" is wrong (medium-low)
- Where: page:160/161.
- Source: docs/review_code_2026-11-04.md:3 and :54. Flow 19 MFLOP per step, 64 steps at sampling = 1.2 GFLOP = 3.5% of one 34.7 GFLOP forward (0.07% of the 50 forwards). In training, the encoder (319 MFLOP) plus the flow add 0.97% of a forward, and plan A trains 66176 steps instead of 66816 to pay for it.
- Also page:160 "a code with a fixed amount of information": the channel bounds the information at R nats (JiT_T2I.code_encode docstring). It does not fix it.
- Correct wording: "The encoder and the flow add 1% to the training FLOPs, which the code run pays with 1% fewer steps, and the flow adds 0.07% to sampling." And "a code that carries at most a fixed amount of information".

### 9. "9.5% lower" is 9.3% (low)
- Where: page:141/142, tc:130, commit eb4dd4c message context. tc:130 also says "1.1% below the crop arm".
- Source: arm_table.py on ladder_crop_1104_1538, unrounded: control 29.464, control2 29.488, crop 26.726, crop with pinning 26.360 ms. The crop is 9.29% and 9.37% below the two controls; arm_table itself prints −9.3%. 9.5% comes from the rounded 26.7/29.5. Crop with pinning is 1.37% below the crop arm, not 1.1%. Both are still under the 3% bar, so the decisions stand.
- Correct wording: "9.3% lower" (both languages); tc:130 "1.4% below the crop arm".

### 10. Recall 0.013 mixes one draw with a mean (low)
- Where: doc 5.6 table, row "1083.94（两组均值）" (doc:387), and item 5 (doc:412: "基线 0.905、0.013、0.248"). Page:176/177 ("Recall rises from 0.0002 to 0.013").
- Source: prompt_a recall 0.0128, prompt_b 0.0118, mean 0.0123. The row labelled as the mean shows precision (0.905) and coverage (0.248) as means, but recall as set 1 (0.013).
- Correct wording: 0.012 in the mean row and in item 5. On the page, either "0.012" (mean) or say "set 1".

### 11. The record's hours for the code runs include their evaluations (low)
- Where: record, rows h-q8 (`hours: 0.968`, `0.974`). The header says "Rows are main training runs only; a row's hours are wall hours … evaluations … count in Other compute."
- Source: job logs gpic-code_w1111.out (18:42:20Z to 19:40:23Z = 0.967 h) and w1112 (19:40:25Z to 20:38:51Z = 0.974 h). Each job also ran 4 evaluations of about 4.6 min each (phase_times.json), about 0.31 h. Config file to final checkpoint is 0.64 h and 0.65 h.
- Why it matters: main compute (101.31 H100-h, which equals 8 × the sum of the rows; the arithmetic is right) holds about 5 H100-h of evaluation. Other compute may count it again. I did not check that.
- Fix: put training-only hours in the rows (about 0.65), or change the header.

### 12. "最后 3 步 1 − t 小于 0.05" counts 3 steps; there are 2 (low)
- Where: doc:329.
- Source: grid 0, 0.02, …, 0.98, 1 (sampling.py:141–143). 1 − t < 0.05 only at t = 0.96 and 0.98. At t = 0.94, 1 − t = 0.06 is not clamped. The doc's own formula (0.06 × 2/3 × 0.6 × 0.6 = 0.0144) uses 2/3 for that unclamped step. The result 0.0144 and doc:437 (x_hat + 0.24 (x_0.94 − x_hat)) are right.
- Correct wording: "从 t = 0.94 起的 3 步里，最后 2 步 1 − t 小于 0.05，被截断。"

## Part B: code read as a hostile reviewer

- noise_share_real.py and eval_noise.sh. The arms differ only in `noise_seed` / `noise_per_prompt` (gpic_jsonl_dataset.py:61: seed = noise_seed + (idx if per prompt else sample_idx)). The per-prompt seeds use the dataset index, so they do not depend on the world size. prompt_b's seeds (100000+) do not overlap prompt_a's. fixed1's tensor equals prompt_a's noise for prompt 1, which is harmless. The 77% is a valid within-caption share (finding 1 covers its use). The official arm reproduces w1053 to 0.0004% on 4 GPUs against 8.
- overfit_code.py at ddcae31 (identical to HEAD). code_of and sample_enc match JiT_T2I.code_encode, the same channel as in training. z_fit is the squared distance to the nearest centre over d(1 − a²); about 1 for encoder z holds by construction, so z_fit_enc carries no information. The z_fit_flow of 1.55 at 64 nats says that flow samples lie between clusters, yet near equals enc_near. enc_match: finding 4. fm is an EMA of single-batch losses logged every 50 steps. The doc's "from 4000 steps the z arms' loss is lower every time" holds at all 3 checkpoints for N = 32 and N = 256.
- posterior_mc.py: the chain update is the DDPM posterior on the linear path. rho² = SNR(s)/SNR(s′) is right, and the chain lands on x_hat. The reading is in finding 7.
- sampling.py at HEAD (0f4d274). `hat = net(...).float()`: the cast from bf16 to fp32 is exact, so the official Euler path's values do not change. `_stochastic_step` is the DDIM/DDPM family with fresh share η²(1 − SNR(t)/SNR(t′)), which is right. The last step returns x_hat in the eta path and in the land path. The fresh noise is reseeded per rank (1000 + LOCAL_RANK); on one node this equals the global rank. The x_T come from their own generators, so the reseed leaves them alone.
- The reading of stoch_eval_0f4d274 against the rule in jobs/eval_stoch.sh ("eta1 below land by more than 5% at both points"):
  - eta1 is 4.6% above land at C0/8 (1177.89 against 1126.19) and 30.9% above at C0/16 (2075.98 against 1586.26). It fails, in the wrong direction.
  - eta05 is 0.9% above land at C0/8.
  - land itself is 3.9% (C0/8) and 3.7% (C0/16) above the Euler per-prompt baselines (1083.93, 1529.61). Removing the leftover 0.0144 noise raises the FD.
  - The run has no Euler control arm, so the land-against-Euler comparison spans two commits (2bd71b7, 0f4d274) and rests on the exact cast. The results appear in no report. Doc 5.7 still lists test 0 as planned.
- z0 from crc32(key) (lightning_model_gpic.py:267): z0 is per prompt, independent of x_T, and the same in the official and the per-prompt evaluations of a checkpoint, so those two differ only in x_T. With 50,000 keys, crc32 has about a 25% chance of at least one colliding pair, which means one pair of prompts shares z0. That is negligible. Plan A has prompt-varying randomness under the official x_T; the page says so ("The right panel favours the code").

## Part C: causal sentences

| sentence | isolated? |
|---|---|
| page:137 "seeds all GPUs alike, so … only 32 different pairs" | by code; not rechecked |
| page:137 "sign changes and one pair gives no noise estimate, so … no clear gain" | yes |
| page:139 "Pinning fails … the node runs out of CPUs"; "So the workers fall behind … a rank waits" | yes: w1094 shows the workers' CPU queue at 93 CPU-equivalents against 4, and q_ready at 0 at the waits; w1117 crop with pinning has no stalls |
| page:139 "The 96 CPUs set a limit near 26.7 ms" | bound only; finding 5 |
| page:141 "Less CPU per image does lower the mean" | yes (w1117, both controls, same node, one run each) |
| page:160 "so nearly all the compute goes to the part that the code leaves open" | design statement; the FLOP figure is wrong (finding 8) |
| page:162 "So the gain appears only at low compute" | no (finding 3) |
| page:164 "A smaller test shows why the gain may be small" | no: an unconditional test on 256 training images, linked to FD on full conditional data; hedged |
| page:164 "So … the flow is not the limit" | yes on this set |
| page:164 "the network hardly uses the code" | no (finding 4) |
| page:170 "so the rule compares both models with one noise per prompt" | design reason, sound |
| page:174 "So all 50,000 prompts start from one noise tensor" | yes (code) |
| page:174 "and the variety … comes only from the captions" | misleading (finding 1) |
| page:176 "so with this protocol the FD hardly depends on the draw" | yes (0.0065%, one pair of 50K draws) |
| doc:295 "多样性只来自 caption。协方差少了噪声贡献的部分" | misleading (finding 1); the removed part is about 6% |
| doc:326 "这符合 z 降低回归目标方差" | consistent, not isolated; worded as consistent |
| doc:329 the leftover noise "正好是官方 Euler 网格加 t_eps 留下的噪声" | yes by arithmetic; the step count is wrong (finding 12) |
| doc:349 "所以新代码路径没有问题" | yes |
| doc:354 "所以大部分是噪声和 caption 的交互" | yes (70% interaction, 6% noise main effect) |
| doc:364 "所以这个模型低估了自己的误差" | partly (finding 7) |
| doc:364 "所以模型前后不完全一致" | yes (gap 2–15 times the Monte Carlo noise) |
| doc:375 "所以混合的图来自主网络，不是小 flow" | yes on this set |
| doc:375 "所以主网络基本没有用 z" | no (finding 4) |
| doc:404 "所以按规则方案 A 没有胜过点模型" | yes |
| doc:410 "所以这个码率下编码部分塌缩"; "所以 review 担心的 EMA 旋转没有出现" | not checked (training logs) |
| doc:413 "方案 A 的收益在低算力处" | no (finding 3); the next clause is hedged |
| doc:431 "所以低算力时直接抽样的噪声偏大" | not isolated; stoch_eval agrees (eta1 +31% at C0/16, +4.6% at C0/8) |
| doc:437 "所以样本对比度低 1.44%，还带 0.0144 的噪声" | yes by arithmetic, for x_hat ≈ x |
| tc:113 "So the median of the pin arm is not sustainable" | yes by arithmetic |
| tc:115 "The 96 CPUs set the limit … Only less worker CPU per image can move the mean" | bound only; "only" is not shown (finding 5) |
| tc:115 "d1280 … far from this limit, so it gains nothing" | yes (crop at d1280: −0.2%) |
| tc:130 "This confirms the last section" | direction only (finding 5) |

## Checked and found correct

- Every FD, fd_mu, fd_sigma, precision, recall and coverage in doc 5.5 and 5.6, the figs CODE array and the noise panels, against metrics.json (except the recall in finding 10).
- Percentages: −27.9% and −27.1% (noise), −23.4% (mean of the shared tensors), shared spread 10.8% (or 9.8% from the top), 0.0065% between per-prompt draws, plan A −8.87/−1.69 and −10.29/+1.86, official-noise −16.45/−17.56 (64 nats), seed pair +2.1/−3.5.
- fig-code caption: "9 to 10% lower … within 2% at an eighth … neither rate passes". It holds at the stated strength. Plan A fails the rule whatever the noise estimate.
- English and Chinese state the same numbers in every audited paragraph and caption.
- Doc 5.4 and the rerun table against the overfit JSONs: all values, "2.4 images" (0.7594 × 256 − 192), the neighbouring covered values 0.906, 0.656, 0.875, and one-call near 0.016 against 0.39.
- Posterior table (doc:360–362): reproduced exactly from the shard sums.
- noise_share.json values: 1013, 1318, 77%, 1264–1278.
- Arm tables: ladder_pin3 and ladder_crop reproduce tc and the page (means, medians, p90, p99, stalls). The crop arm's data waits over 150 ms fall only at steps 1000 and 2000, the save steps, as in every arm.
- tc:113 arithmetic: 8737 images/s, 8.97 ms, 92.6 CPUs, 26.70 ms.
- The plan A baseline has the same shard order (run_code.sh:25 GPIC_SHARD_SEED=1730000000) and the official seeding.
- The official noise is one tensor: gpic_jsonl_dataset.py at 0a1db96 seeds with sample_idx.
- Code link: `gh api repos/wenhaochai/generating-distributions/commits/beb7bbf` resolves (repo public). Its files equal ddcae31's with paths scrubbed: identical blobs for lightning_model_gpic.py, training_code.py, JiT_T2I.py and code_sample.yaml; only the path lines differ in overfit_code.py and eval_code.sh.
- W&B: `reself/gpic-speed/runs/code-r64` and `code-r256` (and lite-ref-e1) return HTTP 200. The API finds them, finished.
- record cost.main 101.31 = 8 × the sum of the row hours; jobs 8.
- The ladder jobs use the crop (ladder_run at eb4dd4c, train_ladder.sh:42 GPIC_REGION_CROP=1).

## Not checked

- The owner's Q&A artifact (claude.ai), and the composition claim in samples.png (the image did not render for me). The pixel numbers support it: the variance over captions with one tensor is 0.050, against 0.137 for a change of tensor.
- The crop's 1.9 against 3.3 ms per image, and the 0.035% / 2-of-255 difference (tools/test_region_crop.py).
- The training-log claims in doc 5.6 items 3–4 (z_var, proj_std, flow_ed, gradient norms).
- The "63% covariance share" (doc:296) and "32 different pairs" (page:137).
- FLOP parity of plan A (I relied on review_code_2026-11-04.md).
- Whether W&B shows the runs to an anonymous reader (I used an authenticated API).
- The record's Other compute, and whether it counts the code runs' evaluations again.
- Note, not a finding: the shared tensors' 10% spread was measured on one model. If the tensor effect differs between models, the ladder's official-tensor FDs carry a model-by-tensor error of unknown size. A per-prompt evaluation of the ladder points would bound it.
