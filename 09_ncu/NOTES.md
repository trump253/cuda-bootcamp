# Softmax Nsight Compute 分析笔记

状态：待学习者采集和填写，以下空格不是已完成的实验结论。命令与指标解释见 [Day 11 任务](README.md)。

## 实验环境与正确性

- 日期：待填写。
- GPU / 驱动 / CUDA / ncu 版本：待填写。
- Git commit / 构建类型 / 编译架构：待填写。
- 同卡其他任务与明显警告：待填写。
- 正确性：V1 / V2 / V3 是否全部 PASS、各自退出码：待填写。
- Profile 条件：FP32，`shape=(128,4096)`、`grid=128`、`block=256`。
- Clock / cache / replay 设置及报告路径：待填写。

## CUDA Event 三轮对照

先填写主要形状 `(128,4096)`；其他形状如有不同趋势，在表后记录。Event `warmup=10`、`iterations=100`，以下延迟单位统一为 ms。

| 版本 | 第 1 轮（ms） | 第 2 轮（ms） | 第 3 轮（ms） | 平均（ms） | 极差 / 平均（%） |
| --- | --- | --- | --- | --- | --- |
| V1 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| V2 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| V3 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

- V2 相对 V1 加速比（V1 / V2）：待填写 ×。
- V3 相对 V2 加速比（V2 / V3）：待填写 ×。
- 差异是否明显大于本轮波动：待填写。
- 其他形状的趋势：待填写。

## Nsight Compute 对照表

不能把 ncu Duration 与 Event latency 混为同一口径；吞吐同名标签也要按单位区分。`n/a` 不可当作 0。

| 指标 | V1 | V2 | V3 |
| --- | --- | --- | --- |
| Kernel Duration（µs） | 待填写 | 待填写 | 待填写 |
| Memory Throughput（SpeedOfLight，%） | 待填写 | 待填写 | 待填写 |
| DRAM Throughput（%） | 待填写 | 待填写 | 待填写 |
| 实际 DRAM 字节吞吐（GB/s） | 待填写 | 待填写 | 待填写 |
| Compute (SM) Throughput（%） | 待填写 | 待填写 | 待填写 |
| Theoretical Occupancy（%） | 待填写 | 待填写 | 待填写 |
| Achieved Occupancy（%） | 待填写 | 待填写 | 待填写 |
| Barrier（cycles/instruction） | 待填写 | 待填写 | 待填写 |
| Long Scoreboard（cycles/instruction） | 待填写 | 待填写 | 待填写 |
| Short Scoreboard（cycles/instruction） | 待填写 | 待填写 | 待填写 |
| Wait（cycles/instruction） | 待填写 | 待填写 | 待填写 |
| Math Pipe Throttle（cycles/instruction） | 待填写 | 待填写 | 待填写 |
| global-load request（request） | 待填写 | 待填写 | 待填写 |
| global-load sector（sector） | 待填写 | 待填写 | 待填写 |
| sector/request（sector/request） | 待填写 | 待填写 | 待填写 |
| shared-load 指令（warp 指令） | 待填写 | 待填写 | 待填写 |
| shared-store 指令（warp 指令） | 待填写 | 待填写 | 待填写 |

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

Review：待进行。
