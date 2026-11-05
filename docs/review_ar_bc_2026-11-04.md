VERDICT: NO-GO（只针对 jobs/run_ar.sh）。自回归模型从来没有在 Lightning 生产路径上跑过，而 jobs/smoke_ar.sh 现在一定会报失败。先修好 smoke 并跑通，再交 run_ar.sh。jobs/run_point_nodrop.sh 可以交。jobs/long_ar.sh 在建好 worktree gpic-wt/ar_long 之后可以交。

独立评审 1c83ca7（ar 分支），2026-11-04。冻结的 worktree gpic-wt/ar_run 在 49d8078。49d8078 只在 1c83ca7 之上改了 jobs/eta_ar.sh 和测试 A 的文件，所以 ar_run 里有这次评审的全部代码。

## 发现（最严重的在前）

1. **自回归模型从未在生产路径上跑过。** jobs/run_ar.sh:20，docs/autoregressive.md:47。
   - 事实：runs/gpic-speed 下没有 ar_smoke_* 目录。logs/ 里没有 gpic-ar-smoke 作业。w1132 到 w1136 都只用 tools/ar/*.py：单卡，没有 Lightning、DDP、CompileWarmup 和 predict_step。review_ar_2026-11-04.md 第 80 行写着：35 ms 每步是估计，没有测量。
   - 失败：run_ar.sh 是第一次让 ARJiT 走 DDP、CompileWarmup、resume、EMA 和 50K 的 predict。任何一处出错，都要等约 20 H100-h 的作业运行后才会暴露。
   - 修复：先修 smoke（第 2 条），用 wq --prio ≥ 10 跑通，再交 run_ar.sh。

2. **jobs/smoke_ar.sh 会误报失败，而且没有测新的部分。** jobs/smoke_ar.sh:39-41。
   - 失败 a：eval_ar.sh 现在写入 eval_000800_ar_base（eval_ar.sh:36）。smoke 检查 eval_000800_ar/metrics.json，所以 `ls` 失败，fail=1。phase_times.json 也读不到。
   - 失败 b：smoke 只跑 base。first64 走 predict_step → ar_sample → `_sample_random(return_order=True)` → redraw，这条路径在 Lightning 中没有跑过。
   - 失败 c：smoke 没有设 GPIC_REGION_CROP=0，所以用 region crop。真实运行用官方 crop。点模型上两者的步时是 29.5 ms 对 26.7 ms（w1117），所以 smoke 测出的步时会偏低约 10%。
   - 失败 d：时间。train 700 s + resume 420 s + 两次评测（timeout 各 2400 s），超过 --time 01:10:00。
   - 修复（要改的全部内容）：
     1. 第 40-41 行改成 eval_000800_ar_base。
     2. 加一次 `eval_ar.sh d640 640 9 2 800 first64`，检查 eval_000800_ar_first64/metrics.json。
     3. 加 `export GPIC_REGION_CROP=0 GPIC_SHARD_SEED=1730000000`，和 run_ar.sh 相同。
     4. 加一行守卫，与 run_ar.sh:15 相同：`grep -q redraw_first` 和 `grep -q "order: random"`。
     5. --time 改成 01:45:00。
     6. 在 phases.log 里记下 BenchTimer 的步时中位数（bench/first、bench/resume），用它定 run_ar.sh 的 --time（第 6 条）。
   - smoke 仍然测这些：训练用 ar_train.yaml，所以 order random 生效（我用 --print_config 确认，见下）。它也测 resume、warm-up 和 warm-up 后的重编译计数。这些部分不用改。

3. **测试 B 读法 3 的 manifold 门槛小于两个权重种子之间的差。** docs/autoregressive.md:373。
   - 事实：w1136（runs/gpic-speed/ar_follow_d5bbd15）里，k = 0 的 manifold 在种子 0 的权重上是 0.674、0.676，在种子 1 的权重上是 0.524、0.524。只换训练种子，差就有 0.15。
   - 失败：long_head6 和 long_few 只和 long_s0 比（同一个种子），门槛是 +0.1。只靠种子噪声就可能过这个门槛。follow 没有这个问题：k = 2 是 0.452 对 0.442，k = 4 是 0.792 对 0.770，种子之间的差 ≤ 0.02。
   - 修复：在运行前从读法 3 中去掉 manifold，或者把它的门槛提到 0.2。另一种做法：在 8 卡作业里再加 long_head6_s1 和 long_few_s1 两臂（6 臂，8 GPU）。

4. **测试 B 的依据数字 12.9% 是"少于 8 个"，不是"8 个以内"。** docs/autoregressive.md:357。
   - 按 training_ar.py:29-32 的规则精确计算：已知 ≤ K ⇔ masked = ceil(256·rate) ≥ 256 − K ⇔ rate > (255 − K)/256 ⇔ |n| < (K + 1)/64，n ~ N(0, 1)，截断在 |n| ≤ 1.2。
   - 已知 ≤ 8：(2Φ(0.1406) − 1)/(2Φ(1.2) − 1) = 0.1118/0.7699 = **14.5%**。
   - 已知 ≤ 7（少于 8）：|n| < 0.125，得 12.9%。所以 12.9% 是"少于 8 个"的比例。如果按 8/256 而不考虑 ceil 来算，也会得到 12.9%。
   - 其他点：已知 ≤ 4 是 8.1%，已知 0 个是 1.6%，已知 ≤ 16 是 27.2%。
   - long_few 臂：已知 ≤ 8 是 0.5 × 0.145 + 0.5 × 9/17 = 33.7%，已知 ≤ 16 是 0.5 × 0.272 + 0.5 = 63.6%。
   - 修复：写成"8 个以内 14.5%"，或者"少于 8 个 12.9%"。结论不变。

5. **测试 B 读法 2 的有效图数没有脚本输出。** docs/autoregressive.md:372。
   - 事实：refine_ar.py 和 follow_ar.py 的 JSON 里都没有样本的有效图数。follow_ar.py 的 effective_all 是第一次抽样的 patch 计数，不是样本的。w1135 的 115、112 是事后从 *_per_sample.pt 的 base idx 算出来的（第 228 行写的范围是 109 到 115）。
   - 失败：一个运行前写定的读法，依赖一个没有写定的事后计算。
   - 修复：在读法中写明：用 `<tag>_refine_per_sample.pt` 里 base 的 idx（2048 个），算各训练图样本数的熵的指数，和 w1135 用同一段代码。或者在 refine_ar.py 里加一个 effective_images 字段，算法和 eval_eta.py:80 相同。

6. **run_ar.sh 的 --time 偏长；步时还没有测量。** jobs/run_ar.sh:5。
   - 训练：52320 步 × 35 ms = 30.5 min，× 45 ms = 39 min。加上编译和 warm-up，约 5 到 10 min。
   - 评测：pred_batch_size 是 32（lite_ladder_sample.yaml 写了两次，后一个 32 生效；--print_config 确认）。每个 rank 6250 张，即 195 个 batch 加一个 10 张的 batch。按 w1133 单卡 batch 32 用 104 s 抽 2048 张：50K 张、8 卡，约 317 s = 5.3 min。真实 caption 比 T = 8 的空 caption 长，text refine 块在每次主干调用中都重算。所以我估计 6 到 7 min。first64 多一次主干调用和 64 × 32 次 head 调用，约多 3%。加上模型加载、Qwen 加载和 fast_eval（点模型约 75 s），每次评测约 8 到 10 min。4 次合计 32 到 40 min。
   - 合计约 1.2 到 1.5 h。02:50 够用，但对 fetch_pending 来说偏大。
   - eval_ar.sh 的 `#SBATCH --time=01:00:00` 在 run_ar.sh 里不起作用，因为它用 bash 内联运行。
   - 修复：smoke 测出步时 ≤ 45 ms、评测 ≤ 10 min 后，改成 02:00:00。否则保持 02:50:00。

7. **对照组与 w1053 之间还有两个差别，都不是 bug。** docs/autoregressive.md:381。
   - a. 去掉 dropout 后，每步少一次 `torch.rand`（training.py:18），所以 t 和噪声的随机数流从第 1 步起就和 w1053 不同。这相当于换了一个训练种子。
   - b. 同一个 shard 种子只固定 shard 的顺序，不固定样本的顺序（logs/ladder_seed.yaml：gpic_dataset.py:270 用 pid 做 hash）。
   - 所以对照组减去 w1053，是 dropout 的效果加上一个种子的噪声（w1053 对 w1054 是 2.1%、3.5%，并且是在官方的共享噪声下测的）。
   - 修复：在测试 C 里加一条运行前写定的读法："对照组对 w1053 的 1529.61、1083.94，两点同向且差超过 5%，才说 dropout 有影响。否则只报告。"现在第 2 项只说"这也测出 dropout 本身的影响"。

8. **eval_ar.sh 的守卫没有检查 mask_token。** jobs/eval_ar.sh:34。
   - 现在不出错：ar_sample.yaml 设了 order random（--print_config 确认），checkpoint 和采样模型都有 mask_token，所以都能加载。
   - 失败场景：将来 overlay 和 checkpoint 的 order 不一致时，strict=False（lightning_model_gpic.py:76）会悄悄丢掉 mask_token 或 bos，另一个保持随机初始化，FD 错误但不报错。head 深度不一致时，多出的 head.blocks.N 也会悄悄缺失。
   - 修复：在第 35 行后加：
     `assert "ema_denoiser.mask_token" in sd, "this checkpoint is not the random order"`
     `assert len({k.split(".")[3] for k in sd if k.startswith("ema_denoiser.head.blocks.")}) == 3`

9. **long_ar.sh 的 worktree 不存在。** jobs/long_ar.sh:13。
   - 事实：gpic-wt/ar_long 不存在。现在交作业，作业会立刻 exit 2。
   - 修复：`git worktree add --detach gpic-wt/ar_long 49d8078`（或 1c83ca7）。

10. **hostile 测试第 18 项的注释算错了。** tools/ar/test_ar_hostile.py:299。
    - 在 N = 16 时，"已知 ≤ 1" ⇔ |n| < 0.5，概率是 0.383/0.770 = 0.497，不是 0.26。few_known_max = 1 时，few 分支的 batch 全部 ≤ 1。所以期望值是 0.5 × 0.497 + 0.5 = 0.75，不是 0.63。
    - 实测 0.82（120 个 batch，SE 约 0.04）。种子固定，所以测试稳定通过。上界 0.85 离期望值只有约 2.5 SE。
    - 修复：注释改成 0.50 和 0.75，范围改成 0.6 到 0.92。

11. **小问题。**
    - run_ar.sh:9 的注释写 133.00 来自 test_ar_hostile.py。实际上来自 tools/ar/test_ar.py 第 7 项；hostile 的第 11 项只核对 head 的 9.87。
    - run_ar.sh:15 的 grep 守卫在 `[ -d ]` 检查之前。worktree 不存在时，报的是 "holds old code"。
    - ar_run 同时服务测试 A（w1140）、测试 C 和对照组。守卫只检查工作区是否干净，不检查 HEAD。所有作业结束之前，不要移动 ar_run 的 HEAD。

## 检查过、结论正确的部分

问题 1：配置合并。
- 在 vis2 上用 `main.py fit/predict --print_config` 解析了真实的参数顺序（train_ladder.sh 的参数在前，run_ar.sh 的 `--config ar_train.yaml` 在后）。
- 训练：denoiser 是 ARJiT，order random，ar_steps 64，head 3 × 1024，eta 0，redraw_first 0，d640、9 + 2 块；trainer 是 ARTrainer，null_condition_p 0.0，few_known_p 0.0，min_mask 0.7；max_steps 52320；W&B 的 name 是 ar_mar_d640，id 是 ar-mar-d640；tags.exp 是 ladder_d640。
- 后出现的 `--config` 只覆盖它写的键，前面的 hidden_size 等键保留。class_path 不变，所以 init_args 合并，不会重置。
- 采样：lite_ladder_sample.yaml + ar_sample.yaml + `--model.denoiser.init_args.redraw_first 64`，得到 ARJiT，order random，ar_steps 64，redraw_first 64；guidance 1.0；bf16-mixed；pred_batch_size 32。
- 点模型：lite_ladder.yaml 单独使用时，null_condition_p 是 0.0。
- 加载：训练和采样两边都是 random order，所以 mask_token 两边都有，strict=False 下正常加载。EMA 是 deepcopy，带着 order、redraw_first 等属性。CLI 的 init_args 先作用于 denoiser，再复制给 EMA。

问题 2：算力。
- random order 下，head 在所有 B × 256 个位置上运行（AR_JiT.py:91，q 是 [B, N, P]），和 raster 相同，所以 head 每张图仍是 9.87 GFLOP（hostile 第 11 项在 vis 节点上重算了 9.87）。
- 主干在两种顺序下都按全注意力计数。mask_token 的 where 和 x_embedder 的差别可以忽略。所以 133.00 和 52320 = 66816 × 104.15 / 133.00 仍然成立。
- 已知 token 上的 head 输出会算出来但不进损失，约占 head 训练 FLOPs 的 13%（rate 的均值是 1 − 0.25 × 0.532 = 0.867），约为总量的 3%。这部分算在 133.00 里，是真实算力。MAR 也是这样。
- 66816：ladder_runs.json 的 d640 C0 是 534400 步；534400/8 = 66800，取 32 的倍数是 66816。
- caption dropout 不改变 FLOPs：dropout 只改 mask，并把 condition 换成同形状的 uncondition（training.py:20-24）。y 的形状仍然是 batch 内最长 caption 补齐到 8 的倍数。SDPA 的 FLOPs 由形状决定。ladder_flops.py 按补齐后的 T（均值 51.5）计数。所以每张图的 FLOPs 改变量是 0。
- ms checkpoint：MS = STEPS/4。52320/4 = 13080，所以在 13080、26160、39240 和 52320 保存。66816/4 = 16704，所以 33408 在里面。

问题 3：采样器。
- predict_step（lightning_model_gpic.py:264-268）对 ARJiT 只调用 net.ar_sample(condition, y_mask)。net 是 EMA，在 fp32 中（on_predict_start）；predict 在 bf16 autocast 下运行。hostile 第 9 项测了 bf16。
- 50000/8 = 6250 张每 rank，DistributedSampler 不补齐。最后一个 batch 是 10 张。代码对 B 没有假设：rank 的 repeat 用的是 y.shape[0]。
- redraw_first 路径：rank.scatter_ 让 order[:, -1] 的秩为 0（先抽的在前）。每行恰好 64 个 pick，满足 redraw "每行个数相同"的前提。从图像 unfold 回来的 token 和抽到的 token 完全相同（已裁剪到 [−1, 1]，unpatchify/unfold 互逆）。context 是 fp32，和 `_sample_random` 相同。y、y_mask 来自同一个 batch，形状是 [B, T, 2048] 和 [B, T]。
- sample_eta：每行个数不同，用 nonzero 取位置，所以可以。y_mask 每行不同也可以，因为 mask 只拼接到注意力里。
- hostile 测试第 1 到 18 项在 vis2 的 CPU 上全部通过（GPIC_AR_NO_COMPILE=1），包括第 18 项的 redraw 调用计数 [[8, 8]] 和 eta 路径的形状。
- 种子：每个 rank 在 batch 0 用 2000 + rank。base 和 first64 用同一个种子，所以每个 rank 的第一个 batch 在 redraw 之前完全相同。

问题 4：对照组。
- run_point_nodrop.sh 和 w1053（smoke_ladder_seed.sh same，代码 1882658，GPIC_SHARD_SEED 1730000000）比：1882658..49d8078 的 diff 里，影响点模型的只有 null_condition_p。JiT_T2I 的新参数（xi_dim、code_dim）都是 0，forward 不变。sampling.py 在 eta 0、land false 时不变。HostTuning 的新开关都没有设置。梯度裁剪在 code_dim 0 时不变。
- GPIC_REGION_CROP=0：run_point_nodrop.sh 导出它，train_ladder.sh:43 用 `${GPIC_REGION_CROP:-1}` 保留 0，gpic_dataset.py 在 import 时读到 0，选 random_crop_arr。1882658 只有 random_crop_arr，所以 crop 和 w1053 相同。
- eval_ladder.sh 的 EVAL_NOISE=prompt 设 noise_per_prompt true，noise_seed 0，50 步，CFG 1.0。这和 w1099 的 prompt_a 相同。差别：8 卡对 4 卡，eval_noise.sh 已测出相差 < 0.1%。
- LADDER_WT 在 train_ladder.sh 和 eval_ladder.sh 两处都生效。ar_run 里的 eval_ladder.sh 有 EVAL_NOISE。
- 目录：runs/gpic-speed/point_nodrop_d640/exp_ladder_d640 和 ar_mar_d640/exp_ladder_d640 都还不存在，不冲突。W&B 项目是 gpic-speed，id 是 point-nodrop-d640 和 ar-mar-d640，resume allow，以前没有用过。
- 时间：w1054 训练加两次评测共 6.14 H100-h，约 46 min。所以 02:00:00 够用。

问题 5：测试 B。
- few_known_p 用 CPU 上的 torch.rand，每次调用抽一次。它只在 overfit_ar.py 中使用（单进程）。ar_train.yaml 不设它，--print_config 显示是 0.0。在 DDP 中，各 rank 的种子相同，所以抽到的值也相同，和 MAR 的 rate 一样。
- 文件：每臂的 tag 不同（long_s0、long_s1、long_head6、long_few），overfit、refine 和 follow 的输出文件名都带 tag，不冲突。
- --head-depth 6 多出 head.blocks.3 到 5。refine_ar.py 和 follow_ar.py 按第 5 个参数建模型，并用 strict=True 加载，所以能加载，深度不符时会报错。
- 时间：w1133 的 random 臂训练 24000 步用 643 s（26.8 ms 每步），抽 2048 张用 104 s，全部（含 rescue）用 810 s。48000 步约 1290 s，加上 7 次各 128 张的检查、2048 张、rescue，约 25 min。head6 的 head FLOPs 翻倍（约 20 GFLOP 对主干 34.5），训练约慢 25%；采样主要是 head 的调用，约慢到 2 倍。所以 head6 约 35 min。refine 和 follow 在 w1135、w1136 中各用 2 到 3 min，head6 约 4 到 6 min，都在 600 s 之内。这个作业约 45 min。"约 1 h"偏多但正确，01:40:00 够用，可以降到 01:15:00。

问题 6：读法。
- 测试 C 读法 1 事先固定 first64，这是公平的：选择发生在运行前，依据是 w1135 的过拟合结果，而且原样的数字一起报告。η 的加测在之后进行，不能替换 first64。文档已经这样写。
- 5% 规则：每个提示一个噪声时，评测噪声很小（prompt_a 对 prompt_b：1083.97 对 1083.90）。剩下的是训练种子噪声。一对种子（2.1%、3.5%，并且在共享噪声下测的）给不出标准差。规则要求两点都超过 5%，所以假阳性的概率低。作为粗略的线，它是合理的。第 2 项已经说明了这一点。
- EMA 的数字：0.9999^26160 = e^−2.616 = 7.3%，0.9999^33408 = e^−3.34 = 3.5%。正确。
- 测试 C 只用 FD 判定，recall 和 coverage 一起报告。没有互相矛盾的读法。

问题 7：smoke。见第 2 条。

## 没有检查的部分

- 没有在 GPU 上运行任何东西。所以没有测量：DDP 下的步时、CompileWarmup 对 ARJiT 是否真的覆盖全部图、warm-up 之后的重编译次数、predict 中 head 编译对动态 M 的处理、真实 caption 下的评测时间。smoke 会测这些。
- 没有读 sample_eta 的离散化，也没有读 redraw 的正确性（另有评审）。
- 没有核对 w1135 的 115、112 是用哪段代码算的。
- 没有核对 W&B 服务器上是否已有同名 id 的运行，只查了本地的 runs/gpic-speed。
- 没有核对 MAR 原文对已知 token 的损失和 head 计算方式，第 2 项算力中的"MAR 也是这样"来自我对 MAR 代码的记忆。

## 修复（2026-11-04）

1. 发现 1、2：smoke 修好了，含 first64、官方 crop、守卫和时限。正式训练等 smoke 通过。步时的中位数由 BenchTimer 的输出读出，再定 `--time`（发现 6）。
2. 发现 3：读法 3 的 manifold 门槛提到 0.2。
3. 发现 4：改成 14.5%。
4. 发现 5：`refine_ar.py` 输出 effective_images。
5. 发现 7：测试 C 加了读法 5。
6. 发现 8：`eval_ar.sh` 加了两个断言。
7. 发现 9、11：测试 B 和测试 C 各用自己的冻结 worktree（`ar_long`、`ar_real`），不移动 w1140 在用的 `ar_run`。
8. 发现 10：注释和范围改了。
