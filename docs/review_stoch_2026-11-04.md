VERDICT: GO — 数学和旧路径都对，没有会改变结论的 bug；下面 3 项应在提交前花 10 分钟修，第 1 项影响结论的读法。

审查对象：commit 99e0c5b（dist-step）。审查人没有提交作业、没有改代码。

## 发现（按严重度排序）

### 1. eta 臂同时改了两件事，读法规则只和 Euler 基线比（中）

- 位置：`baselines/PixelGen/src/diffusion/flow_matching/sampling.py:158-159`，`jobs/eval_stoch.sh:12-15`。
- 问题：eta > 0 时，最后一步 `t_next = 1`，`_stochastic_step` 返回 `hat`。所以每个 eta 臂也做了 land。eta1 和基线 1083.94 的差 = 随机性的效果 + land 的效果。读法规则只拿 eta 臂和基线比。如果 land 本身降低了 FD，eta1 会白白得到这部分收益。
- 修法：作业头部写明“eta 臂也在最后一步落在 x̂ 上”。随机性的效果用 eta 臂对 land 臂来读（同样 5% 门槛）；C0/16 也加一个 land 臂（约 6 分钟），这样两个点都有对照。

### 2. land 和 eta 臂的最终图像是 bf16，基线是 float32（低到中）

- 位置：`sampling.py:159`（`return hat`），`sampling.py:211`（`x = hat`），`sampling.py:163`。
- 问题：predict 在 bf16-mixed autocast 下运行。网络输出 `hat` 是 bf16。官方路径里 `hat - x` 提升为 float32，最终 x 是 float32。新路径把 bf16 的 `hat` 直接当作样本，`vae.decode` 和 `fp2uint8`（`src/models/autoencoder/base.py:30`）都在 bf16 里算。bf16 在 [128, 256) 的间距是 1.0，所以 `(x + 1) * 127.5 + 0.5` 在 bf16 里被舍入，再截断。
- 测量（vis 节点，CPU）：同一个 bf16 的 hat，bf16 路径和 float32 路径的 uint8 结果在 31% 的像素上差 1 级，平均偏高 0.225 级（亮度系统偏高）。另外，`t * hat` 和 `t_next * hat`（0 维 float32 张量乘 bf16 张量）结果是 bf16，每步多一次 2^-9 的相对舍入。
- 影响：很可能远小于 5% 的 FD 门槛，但它是 land 臂和基线之间的一个混杂量。
- 修法：在 `_impl_sampling` 里写 `hat = net(x, t_cur, condition).float()`（或在 `_stochastic_step` 和 land 分支里 `hat = hat.to(x.dtype)`）。eta = 0、land = false 时 `hat - x` 本来就是 float32，所以这个改动不改变旧路径的数值。

### 3. 每个 rank 的注入噪声相同（中，对 eta1_official 更明显）

- 位置：`sampling.py:163`（`torch.randn_like(x)`），`configs_h100/lite_ladder_sample.yaml:3`（`seed_everything: 0`）。
- 问题：Lightning 的 `seed_everything(0)` 在每个 rank 上都调用 `torch.manual_seed(0)`，不加 rank 偏移。4 个 rank 的 CUDA 随机数状态相同，调用顺序和张量形状也相同。所以 rank 0 第 b 批第 j 个样本和 rank 1、2、3 第 b 批第 j 个样本，每一步得到完全相同的新噪声。图像不会重复（提示词不同，noise_per_prompt 时 x_T 也不同），但每 4 个提示词共享全部注入噪声。
- 为什么 eta = 1 时更重要：t = 0 时 λ(0) = 0，所以 fresh = η² = 1。第一步就把 x_T 完全换掉，x_T 只通过 x̂(x_0, t = 0) 起作用。eta = 1 时几乎所有随机性都来自注入噪声。noise_eval 显示“所有提示词共享一个噪声”让 FD 从 1084 升到 1504。4 个一组共享远比这温和，每张图的边缘分布也仍然正确，所以 FD 的偏差应该很小，但没有测过。
- 附带后果：eta1 和 eta1_official 用同一个种子，注入噪声逐位相同，只有 x_T 不同；而 x_T 第一步就被替换。这两个臂预计几乎相同，eta1_official 回答的问题很少。
- 修法（任选一个）：在 predict 开始时 `torch.manual_seed(seed + global_rank)`；或者像 plan A 的 z0 那样，用提示词 key 派生一个 generator，在 `_stochastic_step` 里用它生成噪声（这样每个提示词的结果和 rank 数无关，也可复现）。eta1_official 可以删掉，换成 C0/16 的 land 臂。

### 4. 小问题（不阻塞）

- `sampling.py:155-156` 的 docstring 说“One draw from a Gaussian p(x_t' | x_t)”。实际抽的是 q(x_t' | x_t, x = x̂)。真正的 p(x_t' | x_t) 还多一项 Var(x | x_t) 带来的方差。文档 5.7 的“均值加已知方差”同样如此。建议写成 q(x_t' | x_t, x̂)。
- `sampling.py:162`：eta > 1 时 `1 - fresh < 0`，开方得 NaN。建议在 `__init__` 加 `assert 0 <= eta <= 1`。
- `sampling.py:153`：guidance 是列表时 `float(self.guidance)` 抛 TypeError。但 `_impl_sampling` 第 1 行本来就做 `float(self.guidance)`，所以列表 guidance 在本次改动前就不能用；而且 eta = 0、land = false 时 `and` 短路，不会求值。configs_h100 里所有 guidance 都是标量（4.0 或 1.0）。没有回归。
- `ImplicitSampler`（`sampling_implicit.py:10`）继承 `__init__`，会接受 eta 和 land，但它自己的 `_impl_sampling` 不用它们，静默忽略。目前没有配置这样传参。
- `if t_next >= 1` 在 GPU 张量上触发一次同步，每步一次，开销可以忽略。

## 已检查且正确

1. 数学。路径 x_t = t x + (1 − t) ε，α_s = s，σ_s = 1 − s，λ(s) = (s/(1 − s))²。前向核 x_t | x_t' 的方差是 σ_t² − (α_t/α_t')² σ_t'²。DDPM 后验方差 = σ_t'² (1 − λ(t)/λ(t'))，均值 = α_t' x + sqrt(σ_t'² − 方差) · ε̂，ε̂ = (x_t − t x)/(1 − t)。带 η 时新噪声份额 = η²(1 − ρ²)。代码 `t_next * hat + (1 − t_next)(sqrt(1 − fresh) ε̂ + sqrt(fresh) z)` 与此一致。给定 x 时噪声部分方差 = (1 − t')²，对任意 η ∈ [0, 1] 都保持边缘分布 N(t' x, (1 − t')²)。
2. η = 0：t' x̂ + (1 − t') ε̂ = x_t + (t' − t)(x̂ − x_t)/(1 − t)，即不带 t_eps 截断的 Euler。不过代码里 η = 0 根本不进 `_stochastic_step`（`if self.eta:` 为假），走旧分支。
3. t = 0：λ = 0，ρ = 0，fresh = η²，无除零。最后一步：网格最后一点是精确的 1.0（`torch.cat` 加 1.0，timeshift 1.0 时 shift_respace_fn(1) = 1），在算 λ(t_next) 之前就返回 x̂，不会除零。
4. eta = 0、land = false 时与旧路径逐位相同：唯一的改动是变量名 `out` → `hat`，然后同一行算 `out`。CFG ≠ 1 时 `hat` 未定义，但只在 eta 或 land 打开时才用到 `hat`，而 `__init__` 的 assert 禁止这种组合。`self.guidance` 在 `super().__init__` 里已赋值。没有其它类在 CFG ≠ 1 下继承这段代码（`EulerSamplerJiTAutoGuidance` 继承 `BaseSampler`）。tools/sample_check.py 和 tools/correctness.py 构造 `EulerSamplerJiT` 时不传 eta/land，行为不变。
5. 命令行到达采样器：在 vis2 上打印合并配置，`eta: 1.0`、`land: true`、`noise_per_prompt: true` 都在正确位置；`guidance: 1.0`，`num_steps: 50`，`eval_original_model: false`（用 EMA）。
6. dtype 和设备：`steps` 被移到 `noise.device, noise.dtype`，噪声来自数据集，是 float32。所以 `t_cur[0]` 和 `t_next` 是 CUDA 上的 0 维 float32 张量。x 保持 float32（除第 2 项所述）。
7. predict_step 的闭包：该配置是 `text_mode: masked`，走 `y_mask` 分支。CFG = 1 时 `x.shape[0] == _m.shape[0]`，传入 y_mask。`JiT_T2I.forward` 在 `return_layer=None` 时返回一个张量（unpatchify 的结果）。类型没问题。
8. 作业：两个检查点（ms-step=066816、033408）都存在。`--gres=gpu:4` 和 `--trainer.devices 4` 一致。模型尺寸参数与 eval_noise.sh 相同。fast_eval.py 调用只少了 `--save-features`，不影响 metrics。基线 prompt_a 每臂约 335 秒；5 臂约 30 分钟，在 50 分钟内。eta 的额外计算可以忽略。
9. 基线：prompt_a / prompt_b 在 C0/8 是 1083.97 / 1083.90（均值 1083.93，相差 0.0065%），C0/16 是 1529.61。头部数字正确。协议相同：同一配置、同一检查点目录、noise_per_prompt、50 步、CFG 1.0、4 GPU（批的划分相同）。
10. 预测路径的代码差异：`git diff 2bd71b7 HEAD` 在 baselines/PixelGen 下只改了 sampling.py；其它改动在 docs、jobs、tools/dist_step。`jobs/env.sh` 最后修改于 2026-11-01，早于基线运行（2026-11-04 13:40）。code_run 现在在 ddcae31，它与 2bd71b7 之间也没有 PixelGen 改动。gpic-wt/stoch_run 还不存在；作业在它不存在时以 exit 2 退出，所以必须先在本 commit（或修复后的 commit）创建它。
11. 文档 5.7 的终点：网格步长 0.02。t = 0.94 时 1 − t = 0.06 > 0.05，不截断，残差 x − x̂ 乘 0.04/0.06 = 2/3。t = 0.96 和 0.98 时截断到 0.05，每步残差乘 1 − 0.02/0.05 = 0.6。所以 x_1 − x̂ = 0.36 (x_0.96 − x̂) = 0.24 (x_0.94 − x̂)。若 x̂ 在最后几步等于 x，x_0.94 − x̂ = 0.06(ε − x)，终点 = x + 0.0144(ε − x) = 0.9856 x + 0.0144 ε。结论正确，前提是 x̂ 在最后 3 步不变且等于 x（文档已写“模型认识的图像”）。
12. 文档 5.7 的“数学事实”：给定 x_t，ε 由 x 唯一确定，所以 p(x, ε | x_t) 退化到 x 上；x̃ ~ p(x | x_t) 配 ε̂(x̃) 就是它的精确样本。(x_t, x̃, ε̂) 的联合分布等于 (x_t, x, ε) 的，所以 x_t' = t' x̃ + (1 − t') ε̂ 的边缘分布是 p_t'。逐步归纳，任意步数都精确。前提是 x̃ 来自整张图的真后验。注意：本作业用的是 x̂（均值），不是 x̃，所以这个精确性不适用于本作业；η = 1 的均值代入式 DDPM 在 50 步下通常比 DDIM 差，这是预期内的风险，不是 bug。
13. test_stoch.py 的阈值只在 400 步上断言，50 步只打印。它不检查 bf16 和多 rank（见第 2、3 项）。

## 没有检查

1. 没有在 GPU 上运行任何东西，没有测第 2、3 项对 FD 的实际大小。
2. 没有运行 test_stoch.py（已排除）。
3. 没有检查文档 5.7 引用的表 5.1“每步抽样”一列，也没有检查 5.5 节的数值（0.185、0.132）。
4. 没有检查 fast_eval.py 和参考统计量本身。
5. 没有检查 fetch_pending 主机环境下 `$WC` 和 TMPDIR 的设置（env.sh 与基线相同，假定可用）。


## 修复（2026-11-04）

1. 发现 1：作业头部写明 eta 臂也 land；随机性的效果用 eta 臂对 land 臂读，同样 5% 门槛；C0/16 加 land 臂；删去 eta1_official。
2. 发现 2：`hat = net(...).float()`。
3. 发现 3：eta > 0 时每个 rank 用自己的种子（`1000 + LOCAL_RANK`）。没有按提示词派生噪声。
4. 发现 4：docstring 和文档 5.7 改为 q(x_t′ | x_t, x̂)；加 `assert 0 <= eta <= 1`。
