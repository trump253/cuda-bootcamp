# Day 9：Softmax V0–V3

本节输入、输出均为行主序 FP32 矩阵，形状 `[rows, hidden]`。每一行独立执行
Softmax。按计划依次完成 V0 逐行朴素版、V1 Shared Memory、V2 Warp Shuffle、
V3 合理向量化；每版都要有正确性、Benchmark 与必要的 Profile 证据，不能只凭
源码推断加速。目前仅开放 V0，后续版本在 V0 经 Review 后逐步准备。

## 当前文件

- `v0.cu`：只留下 V0 GPU kernel 的关键 TODO，学习者自行实现。
- `softmax_harness.h`：共用的确定性输入、CPU Reference、GPU 正确性校验、
  CUDA Event Benchmark 和单次 kernel Profile 入口；不用重新写测试样板。
- `DAY9_V0_TASK.md`：当前任务、验收标准和需要提交的结果。

## 计时口径

Benchmark 为同一 GPU 上的 kernel-only CUDA Event 时间：10 次 warm-up、100 次
正式迭代，输出单次平均延迟，单位为 `ms`。内存分配和 H2D/D2H 不计入。
Softmax 的多遍读取与 `exp` 计算让“算法有效带宽”解释不如 Vector Add 直接，
当前框架优先报告 latency；后续分析访存时再结合 Nsight 指标。

当前代码只提供 V0 模板，不能视为完成实现。V0 验收后再记录实测数据与结论。
