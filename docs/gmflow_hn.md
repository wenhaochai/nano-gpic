# GMFlow-HN：给点模型一个方差、融合子步、子步数可调

一句话（2026-11-05）：三项新增都在 CPU 上测过，17 个测试全部通过，9 个变异全部被抓到；GPU 上的两个作业已经写好，没有提交，所以还没有 FD 和吞吐量的数字。

分支 `gmflow-hn`，worktree `gpic-wt/gmflow_hn`，基于 c16d71d。完整英文报告：`docs/gmflow_hn_impl_report_2026-11-05.md`。

## 1. 问题

1. `docs/gmflow.md` 5.3：8 步时 K = 1 的对照已经拿到 K = 8 优势的约三分之二。
2. 这部分优势主要来自 GM-ODE 的子步。子步用学到的方差，在一次网络调用之内更新对干净图的估计。
3. 问题 H：这个优势需要学到的分布吗？点模型加上一个从它自己的误差量出来的方差，是否也能拿到？

## 2. 做了什么

### 2.1 H：点模型的方差表

1. `tools/gmflow/calibrate_spread.py` 测点模型自己的 x 预测误差：s_x(t) = sqrt(E||x − x̂(x_t, t)||² / D)。
2. 数据是 GPIC 验证集 `gpic_val_00000.tar` 的前 4096 张有 caption 的图。训练只读 `data/gpic/train`，所以这是干净的留出集。
3. 图像用验证集的中心裁剪，256 × 256。
4. t 取 [0.01, 0.99] 上均匀的 64 个点。
5. 每张图在每个 t 上用新的噪声。噪声的种子由图的序号决定，所以结果和分片方式无关。
6. 模型用 EMA 权重，fp32 参数，bf16 autocast，没有 CFG，和采样相同。
7. caption 用 Qwen3 的 masked 模式编码，和 `lite_ladder_sample.yaml` 相同。
8. 8 张 GPU 各跑一个进程，图按序号模 8 分开。`merge` 对所有图取平均。
9. 表里还记了每个 t 的标准误和 s_x_trivial = (1 − t) / t（估计量 x_t / t 的误差）。
10. `gm_spread.py` 的 `SpreadTable` 读表，在 t 上线性插值，超出两端取端点值。
11. `GMODESampler(spread_table=...)`：每次调用把 x̂ 变成一个高斯 N(x̂, s_x(t)²)，再做和 K = 1 相同的解析子步。
12. `eval_gmflow.sh` 新增采样器 `gm_ode_cal`：只用于 K = 0，必须给 `EVAL_SPREAD_TABLE`，输出目录是 `eval_<step>_gm_ode_cal_calls<calls>`，表会复制到输出目录。

### 2.2 N：融合子步的 Triton 核

1. 新文件 `gm_ode_triton.py`。一次网络调用的全部 n 个子步在一次 launch 里完成。
2. 每个 program 处理一张图的 256 个像素，K 个分量都放在寄存器里。均值和权重每次调用只读一次。
3. 全部用 fp32 计算。
4. s_x 支持两种形状：每张图一个 (B, 1, 1, 1, 1)，每个像素一个 (B, 1, 1, H, W)。后者是给另一个 worker 的逐像素方差用的。
5. 子步数是编译期常数。每个子步数编译一次。原因：Triton 解释器不能对运行时整数循环。
6. 同一个文件里有同一算法的 PyTorch 参考实现 `gm_ode_substeps_ref`。
7. `GMODESampler(fused=True)` 用这个核。默认 `fused=False`，走原来的 PyTorch 路径，代码不变。
8. `tools/gmflow/bench_sampling.py` 和 `jobs/bench_sampling.sh` 测 8 张 H100 上的采样吞吐量。

### 2.3 G2：子步数可调

1. `EVAL_SUBSTEPS` 覆盖 GM-ODE 每次调用的子步数。默认仍是 ceil(128 / calls)，至少 2。
2. 设置后输出目录加 `_n<substeps>`，不会和默认结果冲突。
3. 只对 `gm_ode` 和 `gm_ode_cal` 有效；其他采样器设置它会报错退出。
4. `evals_gmflow.sh` 的任务可以有第 4 个字段：`step:sampler:calls:substeps`。
5. `evals_gmflow.sh` 可以在任务前加 `--table <json>`，设置 `EVAL_SPREAD_TABLE`。
6. 3 个字段的任务和原来的调用方式不变（T27 测过）。

## 3. 设计上的选择

1. 新逻辑放在新文件里：`gm_spread.py`、`gm_ode_triton.py`、`calibrate_spread.py`、`bench_sampling.py`。
2. `sampling_gmflow.py` 只改了 `GMODESampler`：两个新选项、两个新方法、循环前 3 行。默认值下逐位不变（T22 测过）。
3. s_x = 0 时，`point_mixture` 给出 log s_u = −∞，和原来的 `as_mixture` 完全相同。所以 s_x = 0 的表和不用表逐位相等。
4. 表的插值用 t，不用 σ。理由：任务书要求“在 t 上线性”。
5. 表只给点模型用。给混合模型传表会报错。
6. 测试编号从 T21 开始。T11–T20 留给另一个 worker。
7. 两个作业的 worktree 是常量 `gpic-wt/gmflow_hn_run`。这个 worktree 还不存在，由 owner 冻结后创建。
8. 合并到 `gmflow` 后（2026-11-06），`EVALS_WT` 已删除。`evals_gmflow.sh` 用参数 `--wt <worktree>` 选 worktree，默认仍是 `gmflow_run2`。

## 4. 测试结果（vis1，CPU）

1. T1–T10：全部通过，和改动前相同。
2. T21：插值等于 `numpy.interp`，误差小于 1e-6；超出两端取端点值；`point_mixture` 经 `logstd_x` 还原出 s_x。
3. T22：s_x = 0 的表和不用表逐位相等。和精确一步（ImplicitSampler）最大差 2.4e-7 到 4.2e-7。
4. T23：玩具问题上 x0 是高斯，后验方差已知，网络是精确的后验均值。和精确 ODE 解的 rms 误差：

| 调用次数 | 子步数 | 精确一步 | 真实方差的表 | 用 s_x² 的表（错误） |
|---:|---:|---:|---:|---:|
| 8 | 16 | 0.08551 | 0.00578 | 0.07096 |
| 4 | 32 | 0.16001 | 0.00578 | 0.11277 |
| 2 | 64 | 0.30104 | 0.00577 | 0.16041 |

5. T23 的读法：方差正确时，误差只剩子步的离散化误差，和调用次数无关。精确一步的误差大 15 到 52 倍。
6. T24：在玩具问题上，`error_sums` 恢复出真实的后验标准差，相对误差小于 2%（1500 张图）。3 个分片合并等于一次算完，相对差小于 1e-12。缺一个分片时 `merge` 报错。
7. T25：参考实现等于采样器的 PyTorch 循环（K = 1、3、8，两种 s_x 形状），误差小于 2e-5。
8. T26：Triton 核在 `TRITON_INTERPRET=1` 下在 CPU 上运行，等于参考实现，误差小于 1e-5。像素数不是 block 的整数倍时也对。`fused=True` 等于 `fused=False`。
9. T27：用桩脚本测 shell 逻辑。默认目录名不变；`gm_ode_cal` 必须有表且 K = 0；`EVAL_SUBSTEPS` 加 `_n<n>`；4 字段任务和 `--table` 正确传参；已完成的 `_n` 任务会跳过。
10. 全套用时 31 秒。
11. 变异测试：9 个全部被抓到，包括 4 个新的：

| 变异 | 失败的测试 |
|---|---|
| 表的插值方向反了 | T21、T24 |
| 核里 s_x 平方两次 | T26 |
| `point_mixture` 里 s_x 平方两次 | T21、T23 |
| 核的子步时间差一格 | T26 |

12. 真实检查点的 CPU 冒烟：w1147 的 066816，8 张验证图，2 个分片。s_x 从 t = 0.01 的 0.516 降到 t = 0.99 的 0.055。合并正常。

## 5. 作业（都没有提交）

1. `jobs/calibrate_spread.sh`：8 张 GPU，fetchable，上限 45 分钟。校准 w1147 的 033408 和 066816，输出 `runs/gpic-speed/point_spread_table/<step>/table.json`。预计约 10 分钟。
2. `jobs/bench_sampling.sh`：8 张 GPU，fetchable，上限 50 分钟。点模型官方 Euler 50、16、8、4 次调用；K = 8 的 GM-ODE 8 和 4 次调用，融合和不融合。每个设置至少 60 秒，每张 GPU 1024 个提示词。结果写在 K = 8 的 run 目录旁边。融合和不融合的检查失败时退出 1。
3. 校准之后的评测（需要 owner 决定后再排）：用 `evals_gmflow.sh 0 d640 640 9 2 <root> --link <w1147 dir> --table <table.json> 66816:gm_ode_cal:8 ...`。合并后加 `--wt gmflow_run3`（放在 `--link` 之后），不再用 `EVALS_WT`。

## 6. 风险

1. Triton 核没有在 GPU 上跑过。CPU 上只用解释器测过数学，编译后的数值、速度和寄存器压力要等基准作业。
2. 校准的 s_x 是模型的总误差，不是真实的后验方差。它包含模型自己的偏差。表的值偏大时，子步会过度修正（T23 的 s_x² 行说明方差错了，优势会大部分消失）。
3. 校准误差是整张图的平均。真实的后验方差在不同像素、不同图上差别很大。每张图一个 s_x 可能不够。
4. 校准用验证集的中心裁剪，训练用随机裁剪。分布略有不同。
5. bf16 网络会把 fp32 的舍入差在整条路径上放大。端到端的融合检查用宽松阈值（平均差小于 5e-3，uint8 差大于 1 的像素少于 2%），要在 GPU 上确认这个阈值是否合适。
6. （已解决）`EVALS_WT` 是环境变量，fetchable 作业拿不到。合并后改成参数 `--wt <worktree>`，这个风险不再存在。
7. 和 `gmflow` 分支合并时，`sampling_gmflow.py` 的 `GMODESampler.__init__` 和循环开头可能冲突。改动集中在这两处。
