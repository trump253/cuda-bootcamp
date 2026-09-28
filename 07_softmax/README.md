# Day 9：Softmax V0–V3

本节输入、输出均为行主序 FP32 矩阵，形状 `[rows, hidden]`。每一行独立执行
Softmax。按计划依次完成 V0 逐行朴素版、V1 Shared Memory、V2 Warp Shuffle、
V3 合理向量化；每版都要有正确性、Benchmark 与必要的 Profile 证据，不能只凭
源码推断加速。目前仅开放 V0，后续版本在 V0 经 Review 后逐步准备。

## 当前文件

- `v0.cu`：学习者已实现 V0 逐行串行 GPU kernel；目前保留了原模板的 TODO 注释。
- `softmax_harness.h`：共用的确定性输入、CPU Reference、GPU 正确性校验、
  CUDA Event Benchmark 和单次 kernel Profile 入口；不用重新写测试样板。
- `DAY9_V0_TASK.md`：当前任务、验收标准和需要提交的结果。

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

V0 的正确性、Benchmark、Profile 和基线 Notes 已验收；Day 9 尚未结束，
下一版按计划为 V1 Shared Memory 行内并行规约。
