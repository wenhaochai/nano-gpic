VERDICT: GO. The code is correct: rank, redraw, the metrics and the job all check out. Reading 4's noise rule is weak, and two readings are vague. Fix those (findings 1 to 3) in the docs and the script before you submit. They need no new design.

Review of 0b18b04 and 1729072 (branch ar), 2026-11-04.

## Findings, most severe first

1. **Reading 4 is a poor noise rule, and the script saves nothing that could replace it.** docs/autoregressive.md:196, tools/ar/refine_ar.py:112.
   - Failure: the rule uses one gap between 2 seeds. That gap is |N(0, 2σ²)|. Its median is about 0.95σ, and in about 10% of cases it is below 0.2σ. So "2 × gap" can be anywhere from about 0 to about 4σ.
   - The rule is also unpaired. The two seeds' samples are different samples. But arm against base is a paired difference on the same samples, and its noise is smaller. So the rule is usually too strict, and sometimes far too loose.
   - The JSON holds only means, so you cannot do a paired test after the run.
   - Fix: for each arm, save the per-sample `nn`, `idx`, `best` and `who` to a small .pt file (2048 × 4 values per arm). Then define the noise as a bootstrap over samples of the paired difference (arm − base), for near, covered and near_robust. As a fallback, use the binomial SE: for near ≈ 0.32 on 2048 samples, SE = 0.010.

2. **Reading 2 does not name its table or its bins.** docs/autoregressive.md:194, refine_ar.py:83-85.
   - Failure: the script writes two tables, `stray_by_rank` and `stray_by_rank_near_robust_only`, and gives the first 16 ranks as two bins (0-7, 8-15). The reading says "前 16 个 … 是排在 128 以后的 5 倍", with no table and no bins.
   - In samples that are not near any image, a flag only means "not the image that takes the most patches". In a collage, the correct patches of the second image are flagged too. So in the all-samples table, flags are not stray patches.
   - Fix: write in the reading that it uses the near_robust-only table, and that the rate for ranks 0-15 is the mean of bins 0-7 and 8-15 (the two bins have the same size). Compare that rate with bin 128-255.
   - Also say what 5× means. 16 is 6.25% of 256, so even at 5× only about 25% of the strays are early tokens. Thus a pass does not mean "the strays are the early tokens". It means "early tokens are more often stray". A first16 redraw then cannot remove most strays. If you want to test "most strays are early", also save the share of all stray flags that have rank < 16.

3. **No reading covers the case where covered goes down.** docs/autoregressive.md:197-198.
   - Failure: the model never trained with more than 30% known, and first64 draws 64 tokens at once at 75% known. An arm can make covered worse. Reading 5 covers a rise, and reading 6 covers no change. Nothing covers a fall.
   - Fix: add reading 7. If an arm lowers covered or near_robust beyond the noise, report it. It shows that the model's answers at high known shares are worse than its own draws.

4. **The doc's claim about the last token holds only at 256 steps.** docs/autoregressive.md:189.
   - "最后一个 token 是已知 255 个时抽的" is true at 256 steps. At 64 steps, the last step draws 6 tokens with 250 known.
   - At 64 steps, 10 steps have at least 192 known (75%): 194, 200, …, 250 known. So the claim "the sampler already asks the model at 75% to 94% known" is correct, and it goes up to 98%.
   - But each of those steps draws 6 or 7 tokens. first64 draws 64 tokens in one step, and the sampler never does that at 75% known.
   - The MAR part is correct: training masks at least 70% (review_ar_mar line 82), and sampling runs to 0 unknown.
   - Fix: write "at 64 steps the last step draws 6 tokens with 250 known".

5. **Reading 1 does not say which sampling seed.** docs/autoregressive.md:193.
   - The tolerance itself is fair. The SE of near is 0.010, so the difference of two runs has an SE of 0.015, and 0.03 ≈ 2 SE. For comparison, w1133's training-time sample (another seed) differs from w1134 by 0.016 (covered) and 0.012 (near).
   - Failure: if the rule applies to each of the 2 seeds and to both metrics (4 checks), there is about a 15% to 20% chance of a false stop.
   - Fix: apply it to the mean of the 2 seeds. The mean halves the variance.

## Checked and correct

- **Q1, rank.** `order = keys.argsort`. At each step `after = order[:, :next_left]`, so the step draws order positions [next_left, left). The first step draws `order[:, 255]`. The scatter sets `rank[order[j]] = 255 − j`, so `order[:, 255]` gets rank 0. So 0 = drawn first.
  - Within one step, the ranks are arbitrary, because those tokens are drawn in parallel.
  - At 64 steps, steps 1 to 13 draw 1 token each, so ranks 0-12 are exact. Bins 0-7 and 8-15 are almost exact. Bins from 32 up are wider than one step, so the steps of up to 7 tokens do not weaken the table.
  - `flags` and `rank` both index the raster position.
- **Q2, redraw.** `known=~pick` has the right polarity. The forward replaces unknown positions with `mask_token` after `x_embedder`, so the old picked values do not leak (check 16 tests this with o3 == o1).
  - `z[where]` and `out[where]` use the same `where`, so their order matches.
  - y has batch 32, and 2048 is a multiple of 32, so the batches match.
  - Under autocast, z is cast to float before the head, as in the sampler. The output is float32 and clamped, the same as the sampler's context. `context_noise` is 0.
  - Head batch sizes: the sampler already calls the head with 32 × 1 to 32 × 7 rows. After the second shape, torch.compile marks the size as dynamic, so 512, 1024 and 2048 rows reuse that graph. The backbone shape does not change.
- **Q3, metrics.** `score()` computes the same `nn_rms`, `near`, `covered`, `distinct`, `best_share`, `near_robust`, `covered_robust` and `stray` as eval_saved_ar.py:58-70: the same cdist normalization, thresholds 0.1, 0.2 and 0.9, and the same training images. unfold(unpatchify(x)) == x is exact (I tested it on vis2). So base compares like with like against w1134 (0.523, 0.324).
- **Q4, pairing.** All arms start from the same base tokens. sweep8 passes `current` forward, so each group sees the new values of the earlier groups. That is a Gibbs sweep and matches "依次". The 8 groups partition all 256 tokens.
- **Q7, weights.** random.pt and random_s1.pt both load with strict=True (177 keys, no `_orig_mod.` prefix, pos_embed present). I tested this on vis2.
- **Q7, job.** 4 arms, one GPU each. The output names are `random_seed11.json`, `random_seed12.json`, `random_s1_seed11.json` and `random_s1_seed12.json`, and none of them clash.
  - The guard exits 2 while ar_run is at d9bf244, because that tree has no `def redraw`. So you must move ar_run to 1729072 first. The output directory is then `runs/gpic-speed/ar_refine_1729072`.
  - env.sh sets TMPDIR and the inductor cache. The job needs no network.
  - Time: about 2 min of sampling, plus compile time. Scoring holds a [32, 256, 256, 768] fp32 tensor (6.4 GB), which fits on an H100. Each redraw arm takes 64 to 512 forward passes. A total of about 6 min is plausible. The 20 min timeout per arm and the 25 min job limit are enough.
- **Tests.** test_ar_hostile.py passes on vis2, including check 16.

## Not checked

- I did not run refine_ar.py on a GPU, so its run time and memory are estimates.
- I did not read MAR's own code. For MAR, I relied on docs/review_ar_mar_2026-11-04.md.
- I did not check that ar_diagnose_cabc237 is w1133's output.
- I did not check the threshold 0.7594 or its source.

## 修复（2026-11-04）

1. 发现 1：脚本保存每个样本的数值，并写入配对差和 bootstrap 标准误；读法 4 改用它。
2. 发现 2：读法 2 写明了表和分格，并说明 5 倍的含义；脚本加了 `stray_share_rank_below`。
3. 发现 3：加了读法 7。读法 5、6 也改用同一个噪声。
4. 发现 4：文档改成 64 步时的实际情况。
5. 发现 5：读法 1 改用两个采样种子的均值。
