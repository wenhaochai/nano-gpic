VERDICT: GO（measure()、baseline 和作业都正确；下面的问题只影响读法的措辞，提交前改文档就够，不用改代码。）

评审对象：c505d37（`tools/ar/follow_ar.py`、`tools/ar/test_follow.py`、`jobs/follow_ar.sh`、`docs/autoregressive.md` 跟随测试一节）。2026-11-04。

## 问题（按严重程度排序）

1. 读法 4 可能读到最近图归属的偏差，而不是抽图偏向。位置：`tools/ar/follow_ar.py:78`、`docs/autoregressive.md:275`。
   - 故障：k = 0 的 counts 包括所有抽样，也包括离所有训练 patch 都超过 0.2 的抽样。模型的误差偏向平滑（低频），baseline 的噪声是各维独立的白噪声。平滑的抽样落在“居中”的训练 patch 附近（hubness）。这样 `effective_per_position` 会下降，baseline 不会下降。所以比值低于 0.7 可能只说明误差的形状不同，而不说明偏向。
   - 修复：再存一份只计 `nearest < 0.2` 的抽样（manifold 命中）的 counts，读法 4 用这一份。代码改一行：`add_counts(counts_hit, torch.where(nearest < limit, who, -1))`，或者对 `nearest[pick] < limit` 做掩码。如果不改代码，就在读法 4 里写：比值低于 0.7 且 manifold 低于 0.9 时，不读。
2. 读法 4 缺中间带。位置：`docs/autoregressive.md:275`。比值在 0.7 到 0.9 之间时没有规则。修复：加一句“中间的值只报告”。
3. k = 64 和 128 超出了训练范围。位置：`docs/autoregressive.md:272`。训练时已知最多 30%，也就是约 77 个 token（`AR_JiT.py:144`、`overfit_ar.py:104`）。所以读法 1 的检查（k = 128）问的是训练没见过的已知比例。w1135 在已知 192 个时有效，所以这个检查是公平的。但 k = 128 时 follow 低于 0.9，也可能是分布外的问题，不一定是测量错了。修复：在读法 1 里写明这一点，也报告 k = 64 的值。
4. 在 k = 1、2 时，`ideal` 比准确模型低。位置：`tools/ar/follow_ar.py:75-76`。我在这 256 张图上算了容差 0.2 下的值：
   | k | 一致的图数（均值） | ideal |
   |---|---:|---:|
   | 1 | 6.0 | 0.630 |
   | 2 | 1.5 | 0.919 |
   | 4 | 1.0 | 0.995 |
   | 8、16 | 1.0 | 1.000 |

   一个准确记住 256 张图的模型，给 1 个真实 token 时后验只有这张图本身（patch 两两都不同），所以 ideal = 1。容差的作用是把相近的 patch 算成同一个。这会让 k = 1、2 的 follow/ideal 偏高。读法 3 只用 k = 4、8、16。这些 k 的 ideal ≥ 0.995，所以 0.8 和 0.5 两条规则不受影响。实际上读法 3 就是读 follow 本身。修复：在文档里写明“k ≥ 4 时 ideal ≈ 1，k = 1、2 时 ideal 是一个宽松的下界”。
5. 读法 5 的 `who` 包括不是 near_robust 的样本（`tools/ar/refine_ar.py:86-91`，所有 2048 个样本都有 `who`）。拼贴样本的 `who` 是噪声，会把秩相关拉低。影响小，因为 84% 的样本是 near_robust。修复：说明用哪一份。两种都算更好：全部样本一份，只用 `ok` 的样本一份。

## 检查过、结果正确的内容

1. 测试能失败。我在副本上做了 3 个变异，3 个都被测试抓到：
   - ideal 里去掉 `agree`：k = 1 时 follow 0.75，ideal 0.28，断言失败。
   - follow 用 nearest 代替 own：k = 0 时 follow 1.0，断言失败。
   - effective 去掉 exp：断言失败。
   
   没有改动的副本可以通过测试。
2. measure()：
   - known 来自 `argsort(...)[:, :k]`，每行刚好 k 个。k = 0 时 `close[a][:, c].all(-1)` 对所有图都是 True，所以 ideal 等于均匀选图。
   - own 用 `per[arange, idx]` 取，nearest/who 用 `min(1)` 取，都正确。
   - 已知 token 不变的断言正确。它在 redraw 里成立，因为 `out = tokens.clone()`。
3. 并行抽样（问题 2）：每个位置在给定 z 时独立抽样。所以每个位置的边缘分布就是 p(x_i | 已知)，和顺序抽样的第一次抽样相同。follow、manifold 和 counts 都是逐位置的统计，不需要联合分布。
4. k = 0（问题 3）：所有位置都是 mask_token + 位置编码，输入只差位置编码。在一个位置上，一个 batch 里 32 行的 z 完全相同，只有 head 的噪声不同。这就是要测的边缘分布，所以没有问题。
5. 与真实链的第一步一致（问题 5）：`_sample_random` 的第 0 步有 next_left = floor(256·cos(π/128)) = 255，所以它只抽 1 个 token，其他全部遮住，caption 为空。这和 k = 0 的抽样相同。第一个位置是均匀随机的，所以对位置求和的总次数正好预测“第一次抽样决定整张图”时的最终分布。这个检验本身是对的。它不能区分前 1 次抽样和前 2 到 16 次抽样（这些抽样给定了 1 到 15 个已知 token）。它也不能区分是“第一次抽样偏向”，还是“后面的步骤放大了同样的偏好”。
6. baseline（问题 4）：每个位置 1024 次抽样（4 次重复 × 256 张图），baseline 也是 1024 次（32 × 32）。在 K = 256、n = 1024 时，exp(H) 的有限样本偏差约为 256·exp(-255/2048) ≈ 225。两者的偏差相同，所以比值可以读。在 patch 相近的位置上，均匀选图加随机归属仍然接近均匀，所以 `first_true_gap` 主要让模型这一侧下降（问题 1）。
7. 作业：
   - 显存：distances 的中间张量是 32×256×256×768 float32，约 6.4 GB，峰值约 13 GB。close 矩阵只有 16.7 M bool。head 最多 8192 行，宽 1024。H100 80 GB 都放得下。
   - 编译：head 的行数取 9 种值，32·(256−k)。torch.compile 在第 2 种形状出现时就转为动态形状，不会到 8 次重编译的上限。主干的 batch 固定为 32。
   - 时间：288 次 redraw，对比 w1133 的 4096 次主干调用用了 104 s。加上编译，预计远低于 960 s 的超时和 20 min 的上限。
   - 样本图：rank 正确，`order[:, -1]` 的 rank 是 0，所以 `rank < 64` 每行刚好 64 个 token。
   - guard、文件名、TMPDIR（来自 env.sh）都正确。
   - ar_run 现在在 2d3f170。guard 会以 "holds old code" 停下，所以必须先把 ar_run 移到 c505d37。

## 没有检查的内容

1. 没有在 GPU 上运行 follow_ar.py 的 main()，所以编译和显存只是估算。
2. 没有核对 `null.pt` 和 caption 掩码是否和 w1135 用的一样。我只看了代码，两处写法相同。
3. 没有核对 context_noise 在这些权重里是不是 0。如果不是 0，redraw 的 `_read` 会给已知 token 加噪声。这和采样器的做法一样。
4. 没有检查 w1135 的数字本身。

## 修复（2026-11-04）

1. 问题 1：脚本另存只计真实 patch 的计数，基线用同样多的抽样；读法 4、5 用这一份。
2. 问题 2：读法 4 加了中间带。
3. 问题 3：读法 1 写明了训练范围，并报告 k = 64。
4. 问题 4：读法 3 写明 ideal 的数值和 k = 1、2 不读。
5. 问题 5：读法 5 算两份 `who`，写明这个检验不能区分什么。
