# Day 9：Softmax V0–V3

本节输入、输出均为行主序 FP32 矩阵，形状 `[rows, hidden]`。每一行独立执行
Softmax。按计划依次完成 V0 逐行朴素版、V1 Shared Memory、V2 Warp Shuffle、
V3 合理向量化；每版都要有正确性、Benchmark 与必要的 Profile 证据，不能只凭
源码推断加速。V0 基线已验收，现在开放 V1；V2/V3 暂不提前实现。

## 当前文件

- `v0.cu`：学习者已实现 V0 逐行串行 GPU kernel；目前保留了原模板的 TODO 注释。
- `v1.cu`：学习者已实现每行一个 block 的 Shared Memory max/sum 规约。
- `softmax_harness.h`：共用的确定性输入、CPU Reference、GPU 正确性校验、
  CUDA Event Benchmark 和单次 kernel Profile 入口；不用重新写测试样板。
- `DAY9_V0_TASK.md`：已完成的 V0 任务说明。
- `DAY9_V1_TASK.md`：当前 V1 任务、验收标准和提交内容。

V1 的正确性和首次 Benchmark/Profile 已完成；访存映射仍待一次针对性对照。

## 计时口径

Benchmark 为同一 GPU 上的 kernel-only CUDA Event 时间：10 次 warm-up、100 次
正式迭代，输出单次平均延迟，单位为 `ms`。内存分配和 H2D/D2H 不计入。
Softmax 的多遍读取与 `exp` 计算让“算法有效带宽”解释不如 Vector Add 直接，
当前框架优先报告 latency；后续分析访存时再结合 Nsight 指标。

## V0 基线记录

学习者提交的 17 个正确性用例全部 `PASS`，普通程序退出码为 0；Compute
Sanitizer 的 memcheck 报告 `0 errors`、`0 bytes leaked`，退出码为 0。
输入中包含大正数、大负数、全相等行，以及跨 block 的 `rows=129` 用例。

在 `CUDA_VISIBLE_DEVICES=0`、warm-up 10 次、CUDA Event 迭代 100 次的相同
条件下，学习者整程序暖机后独立运行三轮：

| 形状 `(rows,hidden)` | 三轮 latency（ms） | 平均 latency（ms） |
| --- | --- | ---: |
| `(1,128)` | 0.010783 / 0.010732 / 0.010731 | 0.010749 |
| `(16,512)` | 0.053514 / 0.053554 / 0.053532 | 0.053533 |
| `(128,1024)` | 0.313610 / 0.313297 / 0.313242 | 0.313383 |
| `(128,4096)` | 1.249423 / 1.249796 / 1.253243 | 1.250821 |

学习者对 `(128,4096)` 采集的一次 Nsight Compute Profile：
`gpu__time_duration.sum=1.77 ms`、L1/TEX global-load request `49152`、
L1/TEX global-load sector `1572864`，即 `32 sector/request`。这与源码的
`4 warp × 4096 列 × 3 遍读取=49152 request` 相符。一个 warp 内的相邻线程
负责相邻行；同一轮读取的地址相隔 `4096×4=16384` 字节，每 lane 的 4 字节
读数各占一个不同的 32 字节 sector。理想连续读取 32 个 FP32 元素只需
4 个 sector，因此本例每次 load request 的 L1 请求字节数为理想情况的 8 倍。

三遍循环确实各自发出对输入的 global-load 指令；这不等于每遍都从 DRAM
取回数据，缓存命中也计入 L1/TEX request/sector。`grid=1` 还意味着只启动
一个 block；每个线程串行处理 4096 列，并计算两遍指数。当前三项 Profile
指标不足以分离不合并访存、有限并行度与指数运算各自的耗时贡献。ncu 的
`1.77 ms` 是 Profile 条件下的耗时，不与正常 Event 的 `1.250821 ms` 混比。

V0 的正确性、Benchmark、Profile 和基线 Notes 已验收；Day 9 尚未结束。
V1 由学习者实现后，要在相同 shape、GPU 和构建条件下与此基线对照。

## V1 当前实现：正确，但仍是跨 sector 访存

学习者实现的 V1 使用局部 max、shared-memory tree max、局部指数和、
shared-memory tree sum，再归一化写回。所有 17 个 shape 正确性 `PASS`，
程序退出码为 0；Compute Sanitizer memcheck 报告 `0 errors`、
`0 bytes leaked`，退出码为 0。无元素线程分别贡献 `-∞` 和 `0`，
各轮 barrier 均由整个 block 到达；当前代码未发现明显的同步位置错误。

同一 GPU、相同构建条件下，V0/V1 暖机后交替运行三轮，CUDA Event
warm-up 10 次、正式迭代 100 次：

| 形状 | V0 三轮 latency（ms） | V1 三轮 latency（ms） | V0/V1 平均延迟比 |
| --- | --- | --- | ---: |
| `(1,128)` | 0.010753 / 0.010739 / 0.010766 | 0.004117 / 0.002638 / 0.002744 | 3.396× |
| `(16,512)` | 0.053456 / 0.053504 / 0.053334 | 0.003975 / 0.002704 / 0.002683 | 17.122× |
| `(128,1024)` | 0.313569 / 0.312996 / 0.313401 | 0.004653 / 0.004670 / 0.004797 | 66.570× |
| `(128,4096)` | 1.253339 / 1.253089 / 1.252894 | 0.025068 / 0.025168 / 0.025129 | 49.882× |

前两种较短形状的 V1 首轮偏高，不宜用其单轮数值下结论；大形状的
三轮结果较稳定。`(128,4096)` 的延迟显著下降，证明当前 V1 有性能收益，
但不能把加速归因于合并访存。

学习者一次 Nsight Compute Profile：V1 `gpu__time_duration.sum=39.49 µs`，
global-load request `49152`、sector `1572864`，仍是 `32 sector/request`；
V0 对应为 `49152` request、`1572864` sector。当前 V1 在 `hidden=4096` 时，
`cols_per_thread=16`，同一轮读取的列为 `tid×16+i`，相邻 lane 相隔
`16×4=64` 字节，各占不同 sector。V1 的 request 数可按
`128 block×8 warp/block×16 轮×3 遍=49152` 核对。

因此当前测得的加速与“更多 block 并行、每线程串行工作更少”相符，
但这些数据不能分离各项因素的贡献。下一步只改变列分配：让同一轮的
相邻 lane 处理相邻列，在 max、sum、输出三个阶段保持一致；再做相同的
Correctness → Sanitizer → Benchmark → Profile 对照。目标是亲自验证
`sector/request` 是否下降，而非直接宣称 V1 已完成合并访存优化。
