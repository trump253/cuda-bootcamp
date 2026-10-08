# 最终独立 Softmax 实验记录

状态：未开始；所有待填项由学习者在实际测量后填写，不使用 Day 9 的旧数据代替本轮实验。

## 1. 环境与实现

- GPU 编号、型号、同卡其他任务：待填写。
- CUDA/驱动、CMake、ncu/nsys 版本：待填写。
- 构建模式、CUDA 架构、是否使用 -G/fast-math：待填写。
- 输入：FP32 行主序；Reference 为 double；本轮是否改变数学函数：待填写。
- 三版线程映射、block/grid 和每版改变的内容：待填写。
- Event 计时范围、同步对象、warmup/iterations：待填写。

## 2. 正确性与内存安全

| 版本 | 23 个 shape | 最大绝对/相对误差与行和误差（无量纲） | 程序退出码 | memcheck 错误/泄漏与退出码 |
| --- | --- | --- | --- | --- |
| naive | 待测 | 待填 | 待填 | 待填 |
| shared | 待测 | 待填 | 待填 | 待填 |
| warp | 待测 | 待填 | 待填 | 待填 |

原始输出或保存位置：待填写。

## 3. 同条件三轮 Event

三轮原始输出或保存位置：待填写。三版各四个 shape，按下表自行补齐行；列内均使用 ms，不混入 ncu Duration。

| 版本 | shape | 第 1 轮（ms） | 第 2 轮（ms） | 第 3 轮（ms） | 平均（ms） | 极差/平均（%） |
| --- | --- | --- | --- | --- | --- | --- |
| naive | (128,4096) | 待测 | 待测 | 待测 | 待算 | 待算 |
| shared | (128,4096) | 待测 | 待测 | 待测 | 待算 | 待算 |
| warp | (128,4096) | 待测 | 待测 | 待测 | 待算 | 待算 |

两次优化的逐 shape 加速比与是否超出波动范围：待填写。

## 4. Nsight Compute 对照

采集 shape 固定为 (128,4096)，只启动一次目标 kernel。报告位置、原始输出与警告：待填写。

| 指标 | naive | shared | warp |
| --- | --- | --- | --- |
| GPU Duration（µs，按实际单位换算） | 待填 | 待填 | 待填 |
| Memory Throughput（%） | 待填 | 待填 | 待填 |
| DRAM Throughput（%） | 待填 | 待填 | 待填 |
| Compute/SM Throughput（%） | 待填 | 待填 | 待填 |
| Theoretical/Achieved Occupancy（%） | 待填 | 待填 | 待填 |
| 主要 Warp Stall（注明 cycles/instruction 或实际单位） | 待填 | 待填 | 待填 |
| global-load request（次） | 待填 | 待填 | 待填 |
| global-load sector（个，32 字节/sector） | 待填 | 待填 | 待填 |
| shared-load/store warp instructions（条） | 待填 | 待填 | 待填 |

## 5. 用自己的实现回答

1. 朴素版的 warp 地址分布是什么？线程内连续读取和 warp 合并访问是否是一回事？
2. shared 版如何规约、处理无有效列的线程？每处 barrier 保护什么？
3. warp 版如何选择 mask？跨 warp 结果如何交接，行 max/sum 何时对其他线程可见？
4. 两次优化各想减少什么成本？哪些变化是 Event/Profile 直接观测，哪些只是推断？
5. 哪些 shape 赢、输或收益不稳定？如何避免把 occupancy、吞吐百分比或最大 stall 单独当作结论？
6. 若用 Nsight Systems 复查，能否指出主机 API、H2D/D2H、GPU kernel 与同步？可引用已验收的 Day 10 证据，无需新增系统优化。

回答：待填写。

## 6. 验收交付

- 源码与 Event 核心步骤：待提交。
- 构建、正确性、memcheck、三轮 Benchmark 与 ncu：待提交。
- README 的最终实现/优化总结：待填写。
- 导师最终 Review：尚未进行，不将可编译或单次 PASS 写为毕业结论。
