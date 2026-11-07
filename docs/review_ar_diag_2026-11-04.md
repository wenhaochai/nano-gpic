VERDICT: NO-GO: the frozen worktree gpic-wt/ar_run is at eb807fa, so all 4 arms would run the old script, which ignores every option, and write the same file.

# 独立评审：c219e64（自回归诊断任务），2026-11-04

范围：`AR_JiT.py`、`training_ar.py`、`tools/ar/overfit_ar.py`、`tools/ar/test_ar_hostile.py`、`jobs/diagnose_ar.sh`、`docs/autoregressive.md` 的“诊断”一节。
我没有提交任务，没有改代码，没有 commit。

## 发现（按严重程度排序）

### 1. 阻断：任务会跑旧代码（jobs/diagnose_ar.sh:15, :25）
- 故障：`WT` 默认是 `gpic-wt/ar_run`。它的 HEAD 是 eb807fa（`git -C ar_run log -1`）。
- eb807fa 的 `overfit_ar.py` 用 `sys.argv[1..4]` 读参数，不认识 `--name`、`--context-noise`、`--head-width`。多余的参数被静默忽略。
- 结果：4 个臂都是原设计，没有救援测试。4 个进程都把 `n256.json` 和 `n256.png` 写进同一个目录，互相覆盖。输出目录名是 `ar_diagnose_eb807fa`。任务会“成功”结束，数字看起来正常。
- 修复：提交前执行 `git -C gpic-wt/ar_run checkout c219e64`（该 worktree 只有未跟踪的 `logs/`）。再在脚本里加一行检查，例如 `git -C $WT merge-base --is-ancestor c219e64 HEAD || exit 2`，或检查 `overfit_ar.py --help` 含 `--name`。

### 2. 高：head2048 臂比其他臂长很多，可能撞上 2700 s 超时（jobs/diagnose_ar.sh:5, :25, :29）
- 头的参数：1024×3 是 19.3 M，2048×6 是 135.1 M（我在 vis 节点数的），多 7 倍。
- 每张图的头前向 FLOP：约 9.9 G 变成约 69 G。骨干约 25-30 G。每步总 FLOP 约为原来的 2.5 倍。原来 26 ms/步时约 40% MFU，已接近算力上限。估计训练 24000 步从 704 s 变成约 1500-1700 s。
- 采样由头主导（每个 patch 32 次头调用，1 次骨干调用），且受启动开销限制。块数翻倍，估计采样时间约 2 倍：最后 2048 个样本从 347 s 变成约 650 s。
- 原设计的总时间估计（来自 `ar_overfit_eb807fa/n256.json`）：训练 704 s，4 次检查采样 85 s，最后采样 347 s，救援约 60 s（7 个 k 共 1387 个 patch 步 ×2 batch，约 10.8 个满 batch，每个 5.4 s），head_steps_128 最多 22 s，first_patches 和最后的 32 张图约 10 s，编译约 60 s。合计约 1300 s（22 min）。
- head2048 估计合计约 2500-2700 s。离 `timeout 2700` 很近。超时会杀掉救援部分，可能连最后一行 json 也丢掉。
- 其余 3 张 GPU 在第 22 min 后空闲约 20 min。这违反“4 臂等长”的设计目标。脚本注释“about 25 min”对第 4 臂不对。
- 修复（任选一个）：(a) 提交前在 wq 高优先级跑一个短 smoke（如 500 步）测 head2048 的 ms/步和每 batch 采样秒数，再定时间；(b) 把 `--time` 改为 01:10:00、`timeout` 改为 3900；(c) 第 4 臂跳过救援，只跑 covered（门槛只看它）。

### 3. 中：救援读法的阈值不是数字，64 个样本的噪声和效应同量级（docs/autoregressive.md 诊断第 1、3 条；overfit_ar.py:104-115, :158）
- 每个样本的 late 距离是双峰的：要么落在某张图上（约 0.02-0.05），要么是混合（约 0.25-0.35）。均值的标准误约 0.25×sqrt(p(1-p)/n)。p = 0.5 时，64 个样本约 0.016，32 个样本约 0.022。两个种子在 2048 个样本上的 nn_late 已差 0.024（0.221 和 0.245）。
- 所以小于约 0.05 的差别读不出来。head_steps_128（32 个样本）与 k = 0（64 个样本）之比要差约 0.06 以上才有意义。
- “接近 tf_late（约 0.03）”没有数字边界。参考：N = 32 时一条能工作的链给 nn_late 0.016，tf_late 0.0099，比值 1.6。
- 修复：运行前写定数字带，例如 late_nearest ≤ 2×tf_late 为“救回”，≥ 0.15 为“没救回”，中间为“不确定”。同时保存每个样本的距离，报告 late < 0.1 的比例（双峰下比均值稳得多）。head_steps_128 应和同一臂最后一行的 nn_late（2048 个样本，定义相同）比，不和 64 个样本的 k = 0 比。

### 4. 中：first_patches 可能不能区分（overfit_ar.py:118-128；doc 第 2 条）
- 左上角 16×16 的 patch 信息很少，常是天空或纯色背景。很多真实第一 patch 彼此在 0.1 以内时，混合的抽样也会“near”。
- `first_true_gap` 是诊断本身的有效性条件：中位数低于约 0.2 时，first_draw_near 高不能证明 head 没有抽混合。doc 没写这个条件。
- 修复：在 doc 写明“只有 first_true_gap 明显大于 0.1 时才读 first_draw_near”。更有用的是按位置测（见第 8 条）。

### 5. 低：doc 只读 late_nearest，不读 late_source
- k 小时：late_nearest 低而 late_source 高，说明链干净地换到了另一张图。这也算“救回”，读法一致。
- k ≥ 16 时：late_source 才直接回答“链能否延续给定的图”。k = 192 时两者几乎相等（最近图几乎必是源图）。
- 修复：doc 第 1 条加一句：k ≥ 16 读 late_source；late_nearest 低、late_source 高 = 换图但干净。

### 6. 低：doc 没有写部分改善的读法（doc 第 4 条）
- 只写了 covered ≥ 0.7594 算修复。若某臂 covered 从 0.03 升到 0.3，或 nn_late 降一半，doc 没有预先说怎么读。
- 修复：运行前写一条，例如 nn_late 降到 0.12 以下算“方向对，强度不够”。

### 7. 低：context_noise = 0 时采样的 RNG 流变了（AR_JiT.py:115, :120；overfit_ar.py:97）
- 训练路径：`if net.context_noise` 为假时直接用 `patches`，不调用 randn。和 eb807fa 完全一样。`net.context_noise` 是 Python float，在编译的 forward 之外读，不影响编译。
- 采样路径：`0.0 * torch.randn_like(drawn)` 每个 patch 消耗一次 CUDA RNG。之后 draw_patch 的噪声换了一段流。分布不变，只有逐位复现丢了。检查点的采样也会让之后的训练 batch 换流。
- 影响：没有。eb807fa 没有保存权重，编译下训练本来也不逐位确定，所以 base 臂本来就是一个新种子。base 臂应和 w1132 的两个种子（covered 0.031、0.035）一起看，不当作同一次运行。
- 可选修复：`if self.context_noise:` 包住这两处，零开销且保持旧流。

## 第 5 个问题：更便宜或更好的测试

### 8. 建议加入本任务：按行的误差曲线（几行代码，不加 GPU 时间）
- 最后的 2048 个样本已在内存中。对每个样本取整图最近的训练图，算 16 行 patch 每行的 rms。存成 16 个数（再加 rms < 0.1 的比例）。
- 曲线从第 0 行就高：混合从第一批抽样开始。曲线前低后升：链中途漂移。拐点给出“混合从哪里开始”。这比 first_patches 直接，样本量也大 30 倍。

### 9. 建议加入本任务：保存每臂的最终权重
- 现在什么都不存。任何推理侧的追问（温度、头步数、更多救援样本）都要重训 12-25 min。
- 存 `net.state_dict()`（base 约 0.3 GB，head2048 约 0.8 GB，含优化器更多）。之后的推理测试可用 1 张 GPU 几分钟完成。

### 10. 可考虑：抽样温度
- MAR（Li et al. 2024）在扩散头上用温度，对结果影响很大。混合样本可能来自抽到模式之间的低密度区。
- 便宜的近似：draw_patch 的初始噪声乘以 0.7 或 0.85，k = 0，各 1 个 batch。这只是近似温度，但若 late_nearest 明显下降，就说明问题在抽样的随机性而非条件信息。
- 有第 9 条的权重时，这个测试可以事后做，不必放进本任务。

## 我检查过且认为正确的地方

1. 条件增强的实现（training_ar.py:19-23）：q 由干净 patches 和独立的 eta 生成，和上下文噪声无关。目标是干净的 patches。只有骨干读带噪的上下文。位置 i 的噪声只影响 z_{>i}（输入右移一位，因果掩码）。
2. 采样（AR_JiT.py:108-120）：tokens（输出）保存干净的抽样，context（骨干输入）加噪。每个 patch 的噪声只抽一次，之后不变，和训练一致。前缀也加同样的噪声，返回值不变。
3. teacher_forced（overfit_ar.py:97）用和采样相同 std 的噪声。所以噪声臂的 tf_late 和 nn_late 可比。不同臂之间 tf_late 不直接可比，这是设计上的。
4. 噪声是逐值、逐 token 独立的高斯，固定 std，训练和采样相同。噪声水平固定时不需要把它作为条件输入。
5. 截断：draw_patch 在加噪前截到 [-1, 1]。带噪上下文不截断，训练时也不截断，两边一致。AR_JiT.py:102 的注释“real inputs lie in [-1, 1]”对上下文不再成立，但无害。
6. std 0.3 在 N = 256 下不会让上下文失效：一个 patch 有 768 个值，噪声在比较中平均掉。两张图的 patch 距离约 sqrt(768)×0.3-0.5，噪声对距离差的贡献小很多，识别是哪张图仍然容易。0.1 覆盖 tf 级误差（约 0.033）的 3 倍，0.3 接近混合级误差，两个值的区间合理。局限：真实的漂移是模糊和混合，不是独立高斯。高斯增强不一定覆盖这种偏离，这是读法上的风险，不是代码错误。
7. 第一个 patch 没有上下文，噪声臂无法改变它。若第 8 条显示混合从第 0 行开始，第 2、3 臂预期无效。
8. `net.head_steps = 128`：编译的只是 `self.head`，其 forward 不读 head_steps。tau 以张量传入。不会重编译，也不会出错。
9. 行的切片一致：像素行 192-255 对应 patch 192-255（第 12-15 行）。脚本第 62 行断言了 unfold 的顺序。
10. 救援：k = 0 的空前缀不消耗 RNG（测试 13 断言 k = 0 与 None 相同）。k = 192 时末四分之一全是抽出的 patch。
11. argparse：`256 $OUT 24000 0 --name noise03 --context-noise 0.3` 解析正确（我在 vis 节点验证）。文件名 `base`、`noise01`、`noise03`、`head2048` 各不相同（.json、_rescue.json、.png、.log）。
12. 时间限：`timeout -k 60 2700` 最多 2760 s，加上 env 和 gpu_health，在 3000 s 之内。base 和两个噪声臂约 1300 s，余量足够。
13. 显存：head2048 的参数加 AdamW 约 2.2 GB。头的激活（8192 个 token × 2048 宽 × 6 块，bf16）约几 GB。80 GB 足够。CPU：-c 48，4 个进程各 4 个 OMP 线程加 8 个编译线程，够用。
14. 对抗测试在 vis2 通过（`GPIC_AR_NO_COMPILE=1`，28 s，含新增的第 12、13 项）。

## 我没有检查的地方

1. 没有在 GPU 上测 head2048 的真实 ms/步和采样速度。第 2 条的时间是按 FLOP 和 eb807fa 的计时外推的。
2. 没有测 head_steps_128 的真实耗时（头在每个 patch 步中的占比未知，按上限 4 倍估计）。
3. 没有检查 GPU 上的编译行为（vis 节点关掉了编译）。
4. 没有重查因果掩码、损失和 FLOP 计数（已在 docs/review_ar_2026-11-04.md 排除）。
5. 没有看 w1132 的样本图和每样本分布；第 3 条的双峰是按 doc 的描述（“多是模糊的色块，只有少数清晰”）和 near/covered 推断的。
6. 没有检查 wq 提交方式和 fetch 相关的运行环境。


## 修复（2026-11-04）

1. 发现 1：提交前把 `ar_run` 移到新 commit；`jobs/diagnose_ar.sh` 检查代码版本。
2. 发现 2：head2048 臂已删除（负责人问 MAR 后改了臂：base、noise01、random、random_s1）。
3. 发现 3、4、5、6：读法写进 `docs/autoregressive.md`，带数字；`late_ok` 已加。
4. 发现 7：`_read` 在 context_noise = 0 时不取随机数。
5. 建议 8、9：`row_rms`、`row_ok` 已加；保存最终权重。建议 10（温度）有了权重后再做。
