# Softmax Nsight Compute 分析笔记

状态：学习者已于 2026-10-08 提交三轮 Event 和三份 ncu 输出；导师已完成统计，并通过导入本地报告核对关键指标，没有重新采集。下面记录实测数据，源码解释、瓶颈判断与六个问题仍由学习者填写。命令与指标解释见 [Day 11 任务](README.md)。

## 实验环境与正确性

- 日期：2026-10-08。
- GPU：GPU 0，NVIDIA GeForce RTX 2080 Ti。当前环境复核为 CUDA 11.8、ncu 2022.3.0；采集日志未单列驱动版本。
- 当前复核的 Git commit / 构建类型 / 编译架构：`878d6a5` / Debug / `sm_75`。三个 target 的 CUDA flags 均为 `-g -arch=sm_75 -std=c++14`，无 `-G`。
- 同卡其他任务：采集时状态未记录；提交日志未见采集错误或时钟警告。
- 正确性：每轮每版 19 组均 PASS，三轮共 171 条 PASS，无 FAIL；日志未单列每次程序退出码。之前框架复查的 57 组也全部通过。
- Profile 条件：FP32，`shape=(128,4096)`、`grid=128`、`block=256`。
- Clock / cache / replay 设置：命令未覆盖本机默认值，即 base / all / kernel；三版均为 14 passes。
- 报告路径：`build/day11_softmax_v1.ncu-rep`、`build/day11_softmax_v2.ncu-rep`、`build/day11_softmax_v3.ncu-rep`。三份报告均成功导入，下面关键值与粘贴输出一致。

## CUDA Event 三轮对照

主要形状为 `(128,4096)`。Event `warmup=10`、`iterations=100`；极差 / 平均按 `(最大值 − 最小值) / 平均值 × 100%` 计算，它不是置信区间。

| 版本 | 第 1 轮（ms） | 第 2 轮（ms） | 第 3 轮（ms） | 平均（ms） | 极差 / 平均（%） |
| --- | --- | --- | --- | --- | --- |
| V1 | 0.007497 ms | 0.007463 ms | 0.007475 ms | 0.007478333 ms | 0.455% |
| V2 | 0.006831 ms | 0.006796 ms | 0.006789 ms | 0.006805333 ms | 0.617% |
| V3 | 0.006878 ms | 0.006839 ms | 0.006859 ms | 0.006858667 ms | 0.569% |

- V2 相对 V1 加速比（V1 / V2）：1.098893×，平均延迟降低约 9.00%。
- V3 相对 V2 加速比（V2 / V3）：0.992224×，平均延迟增加约 0.78%，绝对差约 0.053 µs；本轮三次配对均稍慢。
- 差异是否明显大于本轮波动：待学习者解释。三次测量可描述本轮趋势，但不足以给出跨环境的统计保证。

其他形状的三轮平均值：

| shape | V1 平均延迟 | V2 平均延迟 | V3 平均延迟 | V3 相对 V2 的延迟变化 |
| --- | --- | --- | --- | --- |
| (1,128) | 0.003414333 ms | 0.002938667 ms | 0.003142000 ms | +6.92% |
| (16,512) | 0.003905000 ms | 0.003375333 ms | 0.003189333 ms | −5.51% |
| (128,1024) | 0.004976000 ms | 0.004156333 ms | 0.004161667 ms | +0.13% |

本轮 V2 的四个形状均比 V1 快；V3 并非所有形状都退步，也并非所有形状都有收益。本节 ncu 仅采集 `(128,4096)`，不能用它直接解释其他形状的瓶颈。

## Nsight Compute 对照表

不能把 ncu Duration 与 Event latency 混为同一口径；吞吐同名标签也要按单位区分。`n/a` 不可当作 0。

| 指标 | V1 | V2 | V3 |
| --- | --- | --- | --- |
| Kernel Duration（µs） | 11.07 µs | 10.24 µs | 10.40 µs |
| Memory Throughput（SpeedOfLight，%） | 33.22% | 35.87% | 35.63% |
| DRAM Throughput（%） | 33.22% | 35.87% | 35.63% |
| 实际 DRAM 字节吞吐（GB/s） | 189.73 GB/s | 204.87 GB/s | 201.78 GB/s |
| Compute (SM) Throughput（%） | 27.14% | 20.31% | 19.65% |
| Theoretical Occupancy（%） | 100% | 100% | 100% |
| Achieved Occupancy（%） | 43.76% | 43.43% | 44.40% |
| Barrier（cycles/instruction） | 1.34 | 1.10 | 1.83 |
| Long Scoreboard（cycles/instruction） | 6.16 | 6.57 | 8.24 |
| Short Scoreboard（cycles/instruction） | 1.44 | 0.96 | 0.62 |
| Wait（cycles/instruction） | 2.15 | 2.07 | 1.68 |
| Math Pipe Throttle（cycles/instruction） | 0.32 | 0.41 | 0.62 |
| global-load request（request） | 49152 request | 49152 request | 12288 request |
| global-load sector（sector） | 196608 sector | 196608 sector | 196608 sector |
| sector/request（sector/request） | 4 | 4 | 16 |
| shared-load 指令（warp 指令） | 34816 inst | 2304 inst | 2304 inst |
| shared-store 指令（warp 指令） | 18432 inst | 2304 inst | 2304 inst |

辅助上下文：

| 指标 | V1 | V2 | V3 |
| --- | --- | --- | --- |
| 每线程寄存器数 | 42 register/thread | 41 register/thread | 64 register/thread |
| 寄存器容量允许的驻留 block 上限 | 5 block/SM | 5 block/SM | 4 block/SM |
| warp 容量允许的驻留 block 上限 | 4 block/SM | 4 block/SM | 4 block/SM |
| Waves Per SM | 0.47 wave | 0.47 wave | 0.47 wave |

数据核对结论，不替代下面的学习者解释：

- ncu 排序同样为 V2、V3、V1；ncu 中 V2 相对 V1 的加速比为 1.081055×，V3 相对 V2 为 0.984615×。它们与 Event 口径分别计算，没有混用。
- V1 → V2：shared-load 指令减少约 93.38%，shared-store 减少 87.50%；barrier 的归一化比值从 1.34 降至 1.10，但这不是总同步耗时减少 17.91% 的证明。
- V2 → V3：global-load request 减少 75%，总 sector 不变；两版 shared 指令数相同。register/thread 增加，但三版 theoretical occupancy 都是 100%，不能直接断言 V3 因 occupancy 下降而变慢。
- 三版 long scoreboard 都是上述主要 stall 中最大的一项；其比值上升不自动代表绝对等待时间增加，也不能直接解释全部延迟差异。
- raw 输出把这些 stall ratio 的单位显示为 `inst`；本机 `WarpStateStats` section 的图轴定义为 `Cycles per Instruction`，本表按该语义记录，绝不能把 1.34 等值当成 1.34%。说明见 [NVIDIA WarpStateStats 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)。

## V1：Shared Memory Tree

- 直接观察：待填写 Duration、shared 指令、主要 stall、occupancy。
- 对应源码：待填写相关 shared 访问与同步位置。
- 支持的解释与尚不能确认的部分：待填写。

## V2：Warp Shuffle

- 相对 V1 的直接变化与幅度：待填写。
- shared 访问/同步变化是否与性能变化一致：待填写。
- 主要 stall 是否转移，是否有其他成本：待填写。

## V3：float4 向量化

- 相对 V2 的 request、sector、Duration 变化：待填写。
- occupancy 与主要 stall 的变化：待填写。
- 有哪些证据支持或不支持 float4 的收益：待填写。

## 瓶颈判断

- 最可能的限制因素：待填写，可以写“现有数据不足以单一归类”。
- 支持判断的至少两项指标及源码理由：待填写。
- 为什么不能只用 DRAM 或 occupancy 一个数下结论：待填写。
- 哪些原因仍属推断，若需确认应补什么最小验证：待填写，不必先做额外大实验。

## 为什么 V3 赢或输

- Event 是否稳定更快，ncu 排序是否一致：待填写。
- request 减少是否伴随实际 DRAM 字节吞吐/总字节量变化：待填写；仅 GB/s 的升降也受 Duration 影响，不能等同于总流量变化。
- 结论：待填写；允许“无稳定收益”，不得预设向量化一定更快。

## 六个问题的回答与 Review 状态

按任务 README 第 5 节逐项作答；可引用上面的分析，不用重复抄写完整段落。

1. 待填写。
2. 待填写。
3. 待填写。
4. 待填写。
5. 待填写。
6. 待填写。

Review：采集与数据统计检查已完成；学习者的六个回答、源码解释与瓶颈判断尚未提交，Day 11 暂不标记验收通过。最终独立 Softmax 验收仍未进行。
