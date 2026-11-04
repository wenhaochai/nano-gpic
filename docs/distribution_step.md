# 每一步输出一个分布的模型：设计笔记

负责人 2026-11-03 提出方向：「我们需要设计一个每一步输出一整个分布而不是一个速度的model」。这份笔记是活文档。到目前为止只用了数值验证和 CPU 玩具实验，没有用 GPU，也没有在图像上测试。

## 1. 已验证的数学

代码：`tools/dist_step/posterior_identity.py`（16 维高斯混合先验）。记 y = x + σε。

1. 一个后验决定后面所有后验：p(x | y_s, σ_s) ∝ p(x | y_t, σ_t) · N(y_s; x, σ_s²) / N(y_t; x, σ_t²)，其中 σ_s < σ_t。数值误差 3.8e-40（权重）、8.9e-16（均值）。
2. ODE 的时间导数是后验的矩：沿 ODE，dD/dσ = (m3 − Cov·r) / σ³，其中 D 是后验均值，r = y − D，e = x − D，m3 = E[e‖e‖²]。与有限差分的相对误差 9.7e-6。
3. 所以一个精确后验给出整条路径，不需要再调用网络：400 个子步到路径终点的误差 4.8e-6，路径长度 8.61。
4. 反过来，速度只依赖后验均值（Variational Flow Matching 的结论）。均值准确、步数够多时，分布的其余部分没有用。
5. 这个玩具里，精确速度场上的 Euler 误差：1 步 0.966，2 步 0.386，5 步 0.056，10 步 0.013，50 步小于 0.001。

## 2. ladder 怎样计分，这一点决定设计

1. C 只算去噪网络的训练 FLOPs。评测固定 50 步 Euler、CFG 1.0。
2. d640 评测 5 万张约 5e4 × 50 × 35 GFLOP = 8.7e16 FLOPs，是 C0 的 0.6%。所以 ladder 不奖励少步数。
3. 由第 1 节第 4、5 点，分布输出在 ladder 上只有两条路得分：
   a. 同样的训练 FLOPs 学得更好。
   b. ladder 增加一个“采样步数”轴。这需要负责人决定。
4. 文献也支持这个判断：GMFlow 在 NFE ≤ 8 时胜过基线，NFE ≥ 32 时 FID 只是“competitive”；Distributional Diffusion 在 λ = 1 时步数多了反而差于基线；Posterior Flow Matching 只输出对角方差，FID 从 25.42 到 22.28，这个 FID 水平很低。没有文献显示在 50 步上有明显胜出。

## 3. 玩具实验

代码：`tools/dist_step/toy_head.py`。所有臂共用同一个骨干（MLP，宽 256，4 块），10000 步，3 个种子（`mix_nll` 1 个种子），只改最后一层和损失。数据是 8 个 token，x_t = t x + (1 − t) ε，t 服从与配方相同的 logit-normal。

头的定义（d 是 token 维度）：

- `point`：x 预测，v 空间 L2，即 JiT 配方。
- `wiener`：每个 token 一个锚点 μ 和一个方差 s²，输出是高斯先验的后验均值 x̂ = μ + (t s² / v)(x_t − t μ)，v = t² s² + (1 − t)²。损失不变。μ、s² 都由骨干给出。这是最简单的“输出一个分布”，也等于一个学习到的跳连。
- `mix_t`：每个 token K = 8 个锚点、logit 和方差，权重由贝叶斯公式算出。损失不变。
- `mix_multi`：同一个头，再在一次骨干前向上计算 4 个更低噪声级的 L2（x_s 取自 x_t 到 x 的桥）。
- `mix_global`：所有 token 的 logit 和似然先相加再 softmax，所以每张样本只有一个模式选择。
- `mix_global_multi`：同时用全局选择和多噪声级损失。
- `tangent`：一个锚点和 4 个方向，协方差 U Uᵀ，没有各向同性部分。
- `mix_nll`：混合后验，负对数似然。这是 GMFlow 的形式，但我没有实现它的 transition loss，所以不算它的公平复现。

数据：

- `coupled`：所有 token 在同一格（棋盘的 8 格），token 2 维。真实联合模式 8 个。
- `manifold`：一个隐角决定所有 token，16 维里的 1 维曲线。
- `embedded`：同一曲线，每个 token 32 维。流形在 token 内部，最接近图像 patch。

能量距离（real 对 real 的 3 种子均值标准差：coupled 0.0005，manifold 0.0005，embedded 0.0013）：

| 50 次骨干调用，第 10000 步 | coupled | manifold | embedded |
|---|---|---|---|
| point | 0.0370 | 0.0065 | 0.0007 |
| wiener | 0.0014 | 0.0009 | 0.0025 |
| tangent | 0.0124 | 0.0019 | 0.0016 |
| mix_t | 0.0030 | 0.0016 | 0.0072 |
| mix_multi | 0.0027 | 0.0018 | 0.0035 |
| mix_global | 0.0020 | 0.0020 | 0.0181 |
| mix_global_multi | 0.0027 | 0.0026 | 0.0123 |

| 1 次骨干调用，16 个免费子步，第 10000 步 | coupled | manifold | embedded |
|---|---|---|---|
| wiener | 0.2007 | 0.0289 | 1.7607 |
| mix_t | 0.2575 | 0.0360 | 1.0310 |
| mix_multi | 0.1687 | | 0.0912 |
| mix_global | 0.1914 | 0.0341 | 1.9613 |
| mix_global_multi | 0.0077 | 0.0180 | 0.0537 |

| 50 次调用，训练早期 | 第 500 步 | 第 1000 步 | 第 2000 步 |
|---|---|---|---|
| embedded，point | 3.07 | 0.76 | 0.116 |
| embedded，wiener | 0.30 | 0.34 | 0.109 |
| embedded，mix_t | 0.23 | 0.34 | 0.219 |
| coupled，point | 0.70 | 0.20 | 0.118 |
| coupled，wiener | 0.11 | 0.044 | 0.020 |

这些结果说明：

1. 50 次调用时，K = 8 没有比 K = 1 更好。在这 3 种数据上，收益来自跳连，不来自多模式。
2. embedded 到第 10000 步时，point 最好或打平（wiener 和 tangent 与它的差在 1.5 个标准差内），mix_t、mix_global 明显更差。这是主要风险。
3. 一次调用要输出完整联合分布，必须同时有全局选择和多噪声级损失。coupled 上只有一个条件时能量距离是 0.17–0.19，token 一致率是 0；两个条件都有时能量距离 0.0077，一致率 0.93。
4. 曲线数据上一次调用只拿到粗结构（manifold 到曲线距离 0.91，真实 0.04），因为 8 个锚点盖不住连续流形。
5. 能量距离单独看不够。coupled 上 point 的格内散布只有真实值的 0.52，但 99% 的 token 在有效格内；wiener 散布 1.06，但只有 80% 的 token 在有效格内。这像 precision 和 recall 的取舍。真实参考训练（40k 步）是 precision 0.92、recall 0.0001，方向上和 point 一致。这只是提示，不是证据。

我曾说 embedded 上混合头变差的原因是“各向同性方差把噪声漏到流形之外”。这个解释被否定了：`wiener` 也是各向同性，却和 `tangent` 打平。原因还没有找到。

## 4. 设计候选（从便宜到昂贵）

1. `wiener` 头（K = 1）：每个 token 多输出 1 个数，代码约 10 行，FLOPs 基本不变。它检验“训练效率”这一条路。JiT 论文认为高维时没有跳连更好，所以结果可能相反。
2. 全局选择加多噪声级损失（K = 8）：d640 每次前向多 2.0 GFLOP（35 GFLOP 的 +5.8%）。多噪声级只算头，代价很小。高维时全局权重会接近赢者通吃，可以把 token 分组做折中。它检验“一次调用输出联合分布”。
3. 隐式采样器：x̂ = g(x_t, t, ξ)，用严格合适的评分规则（能量分数）或 drifting 训练。它天然是联合的。代价是训练时每个 x_t 要多个样本，除非噪声只进入最后几层。Drifting（Deng 等，2026）一步在 ImageNet 256 上 FID 1.54（潜空间）和 1.61（像素空间），是这个方向最强的基线。

## 5. 测试计划（负责人批准后才开始）

每项的基线都是 ladder 的 d640 在 C0/4 的点，所以要等 w1015 至少到 C0/4，同样的数据、同样的评测。

1. 只评测，约 1 h：在 d640 的 C0/4 检查点上比较 50、100、200 步 Euler 的 FD，并保留 5000 张样本算频谱。如果 200 步和 50 步几乎相同，采样器在这条 ladder 上没有空间。
2. `wiener` 头：d640，C0/4，训练约 1.1 h，评测 10 min。看 FD 和 recall。
3. 全局选择加多噪声级：d640，C0/4，训练约 1.2 h，评测 1、2、4、8、50 次调用，约 1 h。

这些时长是估计。第 2 项有收益时，再上到 ladder 最优 TPP 的 C0；第 3 项在 50 次调用上输而在少调用上赢时，先问负责人 ladder 要不要加步数轴。

## 6. 相关工作

- GMFlow，Chen 等，arXiv 2504.05304。
- Distributional Diffusion Models with Scoring Rules，De Bortoli 等，arXiv 2502.02483。
- Posterior Flow Matching，Ma 等，arXiv 2610.13994。
- pi-Flow，Chen 等，arXiv 2510.14974，ICLR 2026。
- Variational Flow Matching，Eijkelboom 等，arXiv 2406.04843。
- Posterior Augmented Flow Matching，Stoica 等，arXiv 2605.00825：用多个候选目标降低目标方差，FID50K 最多改善 3.4。
- MAR 的 diffusion loss，Li 等，arXiv 2406.11838：小 MLP 头表示每个 token 的分布。
- Generative Modeling via Drifting，Deng 等，arXiv 2602.04770。
- Back to Basics（JiT），Li、He，CVPR 2026。
- 苏剑林，生成扩散模型漫谈（七）（八）：最优方差估计，https://kexue.fm/archives/9245，https://kexue.fm/archives/9246。
- 经典的 patch 高斯混合先验：Zoran、Weiss，EPLL。
- 我做了一次搜索，没有找到“网络预测 patch 混合先验、闭式后验均值头、端到端用去噪损失训练”的论文。这不能证明没有。
