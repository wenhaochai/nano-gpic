VERDICT: GO: 我没有找到会改变任何一臂数字的代码错误；提交前必须把 ar_run 移到 5d7fd7f（否则守卫让作业以 exit 2 停止），并改正下面第 1 到 4 项的文档读法。

# 独立评审：5d7fd7f（MAR 的随机顺序和双向注意力），2026-11-04

范围：`AR_JiT.py`、`training_ar.py`、`tools/ar/overfit_ar.py`、`tools/ar/test_ar_hostile.py`、`jobs/diagnose_ar.sh`、`docs/autoregressive.md` 的“诊断”一节。参考：官方 `mar.py`、`diffloss.py`，MAR 论文 v2 的 HTML 全文，MAR 的 GitHub README，FractalMAR 论文（arXiv 2502.17437）的 HTML。
我没有提交任务，没有改代码，没有 commit。

## 提交前必须做的一步

ar_run 现在在 eb807fa（`git -C .../ar_run log -1`）。守卫（`jobs/diagnose_ar.sh:17`）会发现旧代码，作业以 exit 2 停止。它不会跑错，但会浪费一次排队。命令（登录节点可以执行 git）：

```
git -C /scratch/gpfs/GROUP/USER/project/gpic-wt/ar_run checkout --detach 5d7fd7fd75ecdee99622800974a82e83bced60d6
git -C /scratch/gpfs/GROUP/USER/project/gpic-wt/ar_run log -1 --oneline        # 必须是 5d7fd7f
grep -c -- '--order' /scratch/gpfs/GROUP/USER/project/gpic-wt/ar_run/tools/ar/overfit_ar.py   # 必须 ≥ 1
```

如果先改文档再 commit，就把上面的 SHA 换成新 commit 的 SHA。ar_run 只有未跟踪的 `logs/`，checkout 不会冲突。wq 和 Slurm 里没有正在用 ar_run 的作业（w1132 已结束）。

## 发现（按严重程度排序）

### 1. 中：读法 3（行误差）对 random 臂没有意义（docs/autoregressive.md:104）
- 故障：`row_rms` 和 `row_ok` 按光栅行排列。只有 base 和 noise01 臂按行的顺序生成。random 臂按每张图的随机顺序生成，第 0 行不是“第一批抽样”。按对称性，random 臂的 16 个值预期大致平坦。读法 3 没有限定臂，所以会被拿来读 random 臂。
- 第二个问题：`idx` 是整张图的最近邻。如果光栅链在中途干净地换到了另一张训练图，后面的行相对 `idx` 也会变差。所以“前面好、后面差”不一定是漂移，也可能是换图。
- 修复：在读法 3 写明“只读 base 和 noise01”。再写明“前面好、后面差”要用 k ≥ 16 的 `late_source` 和 `late_nearest` 区分漂移和换图（读法 2 已有这个方法）。

### 2. 中：base 臂不是独立的第三个种子（jobs/diagnose_ar.sh:27，docs/autoregressive.md:107）
- 故障：base 用种子 0。w1132 的 n256 也用种子 0。光栅路径在本提交里不多取随机数：`_read` 在 context_noise = 0 时不取随机数，光栅分支的训练器没有变，`ARJiT.__init__` 的参数创建顺序对光栅不变。所以 base 和 w1132 种子 0 有相同的初始化、相同的数据顺序、相同的 τ 和噪声。只有 GPU 的非确定性让轨迹分开。文档说“base 是新的一次运行，和 w1132 的两个种子一起看”，这会把它当成第三个独立样本。
- 修复（二选一）：把 base 的种子改为 2（`run` 函数里的 `0` 换成参数）；或者在文档里写明“base 是种子 0 的重跑，只用于救援测试，covered 不和 w1132 合并成 3 个样本”。

### 3. 中：“head 宽度不敏感”强于数据（docs/autoregressive.md:85）
- 故障：图 4 的表（MAR-L，400 epoch，3 块，作者自己训练的 KL-16，即表 2 中带 † 的那一行）：宽 256 是 3.47，宽 1024 是 2.85，不用 CFG。FID 差 0.62，即 22%。用 CFG 时是 2.45 对 1.97。这不是“不敏感”。而且这是 16 维 latent token 上的数字。768 维像素 patch 需要多宽的 head，这张表不能回答。
- 第二个问题：2.85 是带 † 的 tokenizer 下的数字。表 1 的默认值 3.50 用公开的 KL-16，head 相同。文档把 2.85 写在 3.50 旁边，没有说明 tokenizer 不同。
- 修复：写成“宽 256 是 3.47，宽 1024 是 2.85（作者自己的 KL-16，16 维 token）；768 维像素上的需求不知道”。

### 4. 中：FractalMAR 的 6.15 不能和本节不用 CFG 的数字并列（docs/autoregressive.md:87）
- 我读了论文 HTML：6.15 是 FractalMAR-H（848M 参数）。附录 A 说他们对类别条件生成扫了 CFG 强度和温度。表 4 没有给出不用 CFG 的数字。本节其他数字都是 MAR 不用 CFG 的 FID。
- 第二个问题：FractalMAR 的最后一级用 256 类 logits 按通道抽 RGB 值，是离散分布，不是 768 维的连续 head。它是“在像素上做随机顺序自回归”的先例，不是“用连续 head 建模像素 patch”的先例。
- 修复：写成“FractalMAR-H（848M）6.15，用了调过的 CFG 和温度；最后一级是每个通道 256 类的离散分布”。负责人说他只读了搜索摘要，所以这一句要改成以论文表 4 和附录 A 为准。

### 5. 低：“和 MAR 的差别”漏了几项（docs/autoregressive.md:89）
文档列了：像素 patch、单一主干、每个 token 1 个 τ、head 大小。漏了下面这些。我标出可能改变结果的项。
1. 采样温度（可能改变结果）：MAR 的 README 写明不用 CFG 时用 `--temperature 0.95`。我们没有温度，等于 1.0。温度低会减少混合，而 covered 的门槛是 0.1 的距离，对混合敏感。
2. head 的类型（可能改变结果）：MAR 是 DDPM，余弦噪声表，1000 步训练，学方差（vlb），采样 100 步。我们是 flow，x 预测，32 步 Euler。
3. 编码器只看已知 token（可能改变结果）：MAR 的编码器（16 层）丢掉被遮挡的位置，只处理 64 个 buffer token 和已知 token；mask token 只进解码器（16 层）。我们的 mask token 在全部 9 层里互相注意。文档说了“没有拆分”，但没说 mask token 因此在整个深度里都存在。
4. buffer token：MAR 有 64 个 buffer token，装类别嵌入，有学习的位置嵌入，也起寄存器的作用。我们没有。条件从 adaLN（常数 t）和每层 attention 的 key、value 里的文本 token 进来。我核对了：每个位置在每一层都能注意到文本 token（`JiT_T2I.py:85-90`，mask 是 `[B,1,1,N+T]`，图像部分全 True）。所以每个位置都拿到 caption。过拟合测试里只有 1 个有效的 null token。
5. head 的位置信息：MAR 在 head 之前给 z 加 `diffusion_pos_embed_learned`。我们没有（`AR_JiT.py:96`）。z 里有位置信息：输入加了固定的 sincos `pos_embed`（我核对了它非零），每层有 RoPE。所以 head 间接知道位置。MAR 没有这一项的消融，我判断影响小。
6. dropout：MAR 的 attention 和 proj 都有 0.1 的 dropout。我们没有。对过拟合测试有利，对以后的真实训练是一个差别。
7. embed 之后的 LayerNorm（`z_proj_ln`）：MAR 有，我们没有。影响小。
8. 优化：过拟合脚本是 lr 1e-4，batch 32，wd 0，不用 EMA，读原始权重，500 步预热。MAR 是 lr 8e-4 在 batch 2048（按它的线性规则，batch 32 对应 1.25e-5），wd 0.02，EMA 0.9999。对四臂的对比是公平的（四臂相同），但它不是 MAR 的配方。
9. 位置嵌入：MAR 用学习的绝对位置嵌入；我们用固定 sincos 加 RoPE。影响小。
- 修复：把第 1 到 4 项加进差别列表。再加一句：random 臂失败只说明“我们这个版本的随机顺序”在像素上失败，不说明 MAR 的方法在像素上失败。

### 6. 低：读法 1 说“主要看 late_ok”，但没有给 late_ok 的噪声带（docs/autoregressive.md:102）
- `late_ok` 来自 64 个样本（`rescue` 的 2 个 batch）。二项分布的标准误最大是 √(0.25/64) = 0.0625。两个 k 之间的差的标准误最大约 0.088。
- 修复：写明“late_ok 的差小于 0.18（约 2 个标准误）不读”。或者把 `rescue` 的 batch 数加到 4（多约 1 分钟）。

### 7. 低：文档说 base 是“表里最差的组合”（docs/autoregressive.md:80）
- 表 1 最差的一行是从左到右 + 因果 + 交叉熵，19.58。Diffusion Loss 的四行里，19.23 是最差的。
- 修复：写成“Diffusion Loss 的几行里最差”。

### 8. 低：作业头部说四个 GPU 都忙到最后（jobs/diagnose_ar.sh:10）
- random 臂每张图跑 64 次主干，光栅臂跑 256 次。w1132 的采样受 kernel 启动限制（GPU 利用率约 55%）。所以我估计 random 臂的采样快约 4 倍：最后 2048 张约 90 s，光栅约 345 s。救援测试里光栅臂约 11 个 batch 当量，random 臂 15 个 batch，但每个快 4 倍。估计 random 臂早结束约 5 分钟。这是估计，我没有在 GPU 上测。
- 影响：两个 GPU 空闲约 5 分钟。远低于 90 分钟的取消线，不影响结果。
- 修复：改头部注释，或者不改。

### 9. 低：守卫只检查一个字符串（jobs/diagnose_ar.sh:17）
- `grep -- '--order'` 能拦住 eb807fa 和 c219e64。它不能证明 `AR_JiT.py` 和 `training_ar.py` 是本提交的版本，也不检查工作区是否干净。
- 修复（可选）：`git -C $WT merge-base --is-ancestor 5d7fd7f HEAD && [ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || exit 2`。

### 10. 信息：fm 在两种顺序之间不能比（overfit_ar.py:164）
- random 臂的 fm 只平均被遮挡的 token，它们的上下文少，条件熵高，所以 fm 预期更高。光栅臂的 fm 平均全部 token，每个 token 都有完整前缀。而且 random 臂每个 batch 的遮挡比例不同，50 步一取的 fm 噪声更大。规则没有用 fm，这一点只是不要在读结果时比较它。

### 11. 信息：random 臂的救援测试不按 base 的意思读
- `prefix = tokens[:, :k]` 是光栅顺序的前 k 个 patch。random 臂里，这等于“上面 k/16 行已知，其余按随机顺序补全”，不是“链的前 k 次抽样”。
- 而且给定 prefix 后，调度仍按 floor(256 cos) 计算。我算了：k = 192 时剩下的 64 个 token 每步抽 1 个；k = 0 时每步抽 1 到 7 个。所以 random 臂不同 k 的并行度不同。
- 读法 1 已经写了“救援测试看 base 臂”。这一点只是不要读 random 臂的 `late_*_k*`。

## 我核对过、结论是正确的

### 问题 1：和 MAR 的逐行对照
1. 遮挡比例：`1 - 0.25|N(0,1)|`，拒绝小于 0.7 的值（`training_ar.py:28-30`），和 `stats.truncnorm(-1.2, 0, loc=1, scale=0.25)` 同分布。我各抽 20 万个：均值 0.8668 对 0.8673，KS 检验 p = 0.116。
2. 遮挡个数：`ceil(N * rate)`，和 MAR 相同。
3. 顺序：每张图一个新的均匀排列（`rand(B,N).argsort`），和 MAR 的 `np.random.shuffle` 同分布。遮挡的是顺序里的前 `masked` 个，和 MAR 相同。
4. 损失只算被遮挡的 token：`(per_token * counted).sum() / counted.sum()`，和 diffloss 的 `(loss * mask).sum() / mask.sum()` 相同。已知位置的权重是 0，梯度是 0。`training_ar.py:27` 的注释“as MAR”对这三项是真的。
5. 余弦调度和截断：我用 N = 256、64 步，对 k = 0、1、4、16、64、128、192，逐步比较了我们的 `_sample_random` 和 MAR 的 `sample_tokens` 每步抽的个数。全部一致，总数都是 N − k。最后一步 `next_left = 0` 等于 MAR 的 `mask_to_pred = mask`。`next_left >= left` 时跳过，只在剩 1 个 token 时发生，等于 MAR 的空抽样。
6. 模型看到的东西：已知位置是 patch 嵌入加 `pos_embed`，未知位置是 mask token 加 `pos_embed`（`AR_JiT.py:89`）。全部位置互相注意。
7. 每个位置都拿到 caption（见发现 5 第 4 项）。

### 问题 2：泄漏和错误
1. 未知 token 不到达任何输出：`torch.where` 在 `x_embedder` 之后替换掉未知位置，前向值和它无关，反向梯度是 0。hostile 第 14 项用梯度验证了这一点：未知位置的梯度全是 0，已知位置全大于 0。head 是逐行的 MLP，位置 i 的 q 不进入位置 j。
2. 已知位置不进入损失：权重是 0。head 在已知位置也算了，约 13% 的 head 计算是浪费，不影响结果。
3. 每行的未知个数相同：初始 unknown 是顺序的前 N − k 个（prefix 的 key 加 2，排在最后）。每步 `after = order[:, :next_left]`，`next_left ≤ left − 1`，所以 after 总是 unknown 的子集，每行恰好 `next_left` 个。所以 `int(unknown[0].sum())` 对每行都对。
4. prefix 不会被抽到：它的 key 在 [2, 3)，其他在 [0, 1)，它在顺序的最后 k 个。`next_left ≤ N − k − 1`，所以 after 不包含 prefix，unknown 也不包含。hostile 第 14 项验证了 prefix 原样返回，抽的个数是 2 × (N − 5)。
5. 写回位置：`z[where]` 和 `tokens[where]`、`context[where]` 用同一个 `where`，行优先顺序相同。
6. DDP：`known` 和比例改变的是内容，不改变形状。全遮挡的 batch 里 `x_embedder` 的梯度是 0 但不是 None（hostile 第 14 项），DDP 不会卡住。每个 rank 的比例不同时，每个 rank 的损失是自己的平均，DDP 平均梯度。比例在 rank 之间不同或相同都不会让 DDP 出错（MAR 的做法取决于 `main_mar.py` 的种子设置，我没有读）。注意：生产路径里 `host_tuning.py:121` 在预热后才给每个 rank 不同的种子，预热期间 8 个 rank 抽相同的比例和顺序。这不影响本次单卡作业。
7. 全遮挡的 batch：rate > 255/256 时出现，概率 1.6%（我们 1.59%，MAR 1.62%）。和 MAR 相同。

### 问题 3：编译
1. 训练的形状固定：B = 32，N = 256。`self.order` 是常数属性，dynamo 只追踪一条路径。
2. 编译后的 head：random 臂里每步的行数 M 是 32 到 224。我在 CPU 上用 M = 32、64、…、224、8192 调用 `torch.compile(TokenFlowHead)`：`unique_graphs = 2`。第二个形状触发一次动态形状的重编译，之后不再重编译。不影响结果，只多一次编译。
3. `torch.where` 用 bf16 的 x 和 `.to(x.dtype)` 的参数：hostile 第 14 项的编译路径和不编译的误差 < 1e-3（CPU）。我没有在 GPU 的 autocast 下测编译图。
4. 以后 8 卡 DDP：random 路径的 mask 是 `[B,1,1,N+T]`，和官方 `JiT_T2I.forward` 的 `full_mask` 形状相同；`y_mask = None` 时 mask 是 None。生产配置不设 `order`，默认是 raster，所以本提交不改变现有的生产路径。

### 问题 4：指标
1. `row_rms`：`((g − imgs[idx])²).mean(dim=(1,3))` 得到 [样本, 256 像素行]，`unfold(1,16,16).mean(-1).sqrt()` 先平均平方再开方，得到每个样本每个 patch 行的 rms。然后对样本平均。`row_ok` 是每个样本每行 < 0.1 的比例。形状和计算都对。
2. 最后四分之一：像素行 192 到 255 等于 patch 192 到 255。k = 192 的 prefix 正好覆盖 patch 0 到 191。
3. `teacher_forced` 的 random 分支：约 30% 已知（伯努利，不是固定个数），只在被遮挡的末尾 token 上算 rms，`clamp_min(1)` 防止除 0。
4. `first_patches` 的 random 分支：什么都不知道时位置 0 的分布。含义成立：它是“没有上下文时一个 patch 的边缘分布”。

### 问题 6：作业
1. 臂：base、noise01、random（种子 0）、random_s1（位置参数种子 1，`overfit_ar.py 256 $OUT 24000 1`）。种子 1 同时改变 CPU 上的比例抽样和 GPU 上的顺序。
2. 文件名：每臂 `<tag>.json`、`<tag>_rescue.json`、`<tag>.pt`、`<tag>.png`、`<tag>.log`，互不冲突。输出目录 `ar_diagnose_<ar_run 的 SHA>`。
3. 权重：128.5 M 参数，fp32 0.51 GB 每臂，4 臂共约 2.1 GB。/scratch/gpfs 剩 45 TB。
4. 时间：w1132 的 n256 用了 1057 s（训练 704 s，采样 347 s）。加上救援测试，光栅臂估计约 20 分钟。每臂 `timeout 2700`，作业上限 50 分钟，够用。
5. 作业是 4 GPU、不到 3 小时，按规则走 wq，不需要 `--partial`。

### 问题 7：文档里核对为正确的数字
- 表 1（Diffusion Loss，不用 CFG）：19.23、13.07、3.43、3.50，正确。
- 表 2：KL-16 3.50，VQ-16 7.82，正确。
- 深度“观察类似”，没有表，正确。
- 默认 64 步，正确。

### 测试
在 vis2 上：`test_ar_hostile.py` 退出码 0（14 项全过）；`GPIC_AR_NO_COMPILE=1 test_ar.py` 退出码 0。

## 我没有核对的

1. GPU 上的任何运行：random 臂的实际采样时间、GPU autocast 下的编译图、显存。发现 8 的时间是按启动次数估计的。
2. MAR 的 `main_mar.py` 和 `engine_mar.py`（grad clip、学习率调度、EMA 的代码）。我只读了 `mar.py`、`diffloss.py` 和 README。
3. w1095（点模型）的 covered 是否也用 8N = 2048 个样本。我按文档的说法接受。
4. 读法 1 里“64 个样本的噪声约 0.016”这个数。
5. 用 Lightning 的生产路径跑 random 顺序（没有配置，也不在本次范围）。
6. FractalMAR 的全文。我只读了表 1、表 4 和附录 A 的摘录。


## 修复（2026-11-04）

1. 发现 1：读法写明只读 base 和 noise01，并用 `late_source` 区分漂移和换图。
2. 发现 2：base 和 noise01 用新种子 2。
3. 发现 3、4、5、7：文档已改。
4. 发现 6：救援测试加到 128 个样本，`late_ok` 的噪声带写进读法。
5. 发现 8：作业头部已改。
6. 发现 9：守卫检查 3 个文件和工作区是否干净。
7. 发现 10、11：写进读法。
