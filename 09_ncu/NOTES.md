# Softmax Nsight Compute 分析笔记

状态：学习者已于 2026-10-08 提交三轮 Event、三份 ncu 输出和前五个问题的初步回答；导师已完成统计与答复 Review，并补充 warp stall 的最少理论。下面记录实测数据和 Review，最终瓶颈解释与 stall 理解检查仍待学习者完成。仅导入已有报告，没有重新采集或修改 kernel。命令与指标解释见 [Day 11 任务](README.md)。

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
- 学习者结论：向量化不一定带来稳定收益。Review：正确；补充本轮主要形状三次都稍慢，不能直接说差异全是噪声，也不能凭三次测量推广到所有环境。其他形状已有正收益，结论需限定形状与测量条件。

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
| 活跃周期内平均驻留 warp 数 | 14.00 warp/SM | 13.90 warp/SM | 14.21 warp/SM |
| Issue Active（per_cycle_active，比值） | 0.24 | 0.24 | 0.22 |

后两行来自本次已有报告的 `sm__warps_active.avg.per_cycle_active` 与 `smsp__issue_active.avg.per_cycle_active`，未新增采集。Issue Active 的值折合约 24%/24%/22%，描述对应活跃周期内 scheduler 发出指令的周期比例，不是全 GPU 的繁忙时间比例。

### `68 SM × 4 block/SM = 272 block` 的来源

本次再次导入已有 V2 报告核对，未启动 kernel 或重新采集：

| 数值 | 含义 | 来源 |
| --- | --- | --- |
| 68 SM | 本卡的 SM 数量 | `device__attribute_multiprocessor_count=68` |
| 32 thread/warp | 一个 warp 的线程数 | `device__attribute_warp_size=32` |
| 32 warp/SM | 本架构每 SM 的最大驻留 warp 数 | `device__attribute_max_warps_per_multiprocessor=32` |
| 256 thread/block | 本次 kernel 的 block 大小 | 源码常量与 `launch__block_size=256` |
| 4 block/SM | 按 warp 容量推导的本次驻留上限 | `32 warp/SM ÷ (256/32 warp/block)`，报告 `launch__occupancy_limit_warps=4` |
| 128 block | 本轮实际 grid 大小 | 一个 block 处理一行，`rows=128`；`launch__grid_size=128` |

V2 报告的 block 数、寄存器与 shared 容量限制分别为 16/5/128 block/SM，连同 warp 限制取最小值为 4 block/SM。硬件并非对所有 kernel 都只能驻留 4 个 block；本次 4 个取决于 block 大小与资源需求。于是全卡理论驻留容量为 `68×4=272 block`，实际只有 `128/272≈0.47 wave`。272 是填满当前配置理论容量所需的 block 数，不是 grid 必须设置的数，也不保证一到 272 就能精确测到 100% achieved occupancy。硬件上限参考 [NVIDIA Turing occupancy 说明](https://docs.nvidia.com/cuda/archive/11.8.0/turing-tuning-guide/index.html#occupancy)。

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

1. 学习者：“向量化不一定会带来稳定的收益。”Review：方向正确；需结合三轮波动限定结论，区分主要形状的轻微回退和其他形状的收益。
2. 学习者：“shared-load/store 次数大幅下降支持收益；barrier 变化不大是否证明同步影响不大？”Review：前半正确，但指标精确地说是 warp 级 shared 访存指令数；配合 Event 变快支持优化，而非仅靠计数下降证明速度。后半不能成立：本次 barrier 是按发出指令归一化的比值，不是同步耗时。源码中 V1 有 `2×(1+8)=18` 个 block barrier 阶段，V2 有 4 个；shared 与 barrier 同时改变，现有对照没有独立分离二者的收益贡献。
3. 学习者：“每个 warp 从 FP32 变成 float4，sector/request 从 4 变成 16，request 降为四分之一，总数不变。”Review：正确；更精确地说是每个 lane 一条加载从 4 字节变成 16 字节，满 warp 覆盖量从 128 字节变为 512 字节。本次 `49152×4=12288×16=196608 sector`，这是 L1/TEX 请求覆盖量，不是 DRAM 总流量。
4. 学习者：“不能只看 DRAM；还有同步等开销。”Review：正确；再区分访存带宽限制与访存延迟/依赖等待。带宽未满也可能在等数据，不能反过来直接断言 compute-bound。
5. 学习者：“实际利用率约 44%，可能因为任务短、并行程度不够。”Review：方向正确，但这个数是驻留 warp 占 SM 容量的比例，不是全卡利用率；等待数据或 barrier 的驻留 warp 也计入 occupancy。本卡 68 个 SM、每 SM 最多 32 个 warp，每 block 8 个 warp，理论容量允许 4 block/SM；填满这些容量需 272 个 block，本轮只有 128 个。`Waves Per SM=0.47` 支持工作量不足一整波的解释；短 kernel 的起止阶段还可能影响平均值，但“运行时间短”本身不必然导致低 occupancy。
6. 学习者询问：“stall 具体是什么，要结合哪些参数分析？”导师讲解：见下面的方法与 [复习问答](../CUDA_复习与面试问答.md#11-nsight-compute-单-kernel-分析)。本题尚未由学习者独立解释本轮瓶颈，不标记通过。

## Warp stall 的最少分析方法

stall 是某个 warp 的下一条指令暂时不能发出，例如要用的数据尚未返回、还没等齐 block 同伴或目标执行管线暂时不能接收指令。该 warp 等待时，scheduler 可以给其他就绪 warp 发出指令；因此一个 warp 等待不等于整张 GPU 停住，也不等于其不再计入 occupancy。

| 原因 | 先理解为 | 本节配合查看 |
| --- | --- | --- |
| Long Scoreboard | 等 L1/TEX 路径上的访存结果依赖，不一定等于 DRAM miss | global request/sector、L1/L2 hit rate、DRAM 吞吐、驻留 warp 与 Issue Active |
| Short Scoreboard | 等 shared 等非 L1/TEX 操作的结果依赖，不一定是 bank conflict | shared-load/store 指令、相关依赖；若判断 bank conflict，要有专门计数器证据 |
| Barrier | 已到 block 同步点，等同 block 其他 warp | barrier 所在位置与执行阶段数、同步前工作分配，而非只数函数调用 |
| Wait | 等固定延迟执行依赖 | 源码的数据依赖和 Issue Active，不能把它全部归到 expf |
| Math Pipe Throttle | 所需数学执行管线暂时不能接收指令 | 相关管线吞吐与指令混合，而非只看 SM 汇总百分比 |

定义参考 [NVIDIA warp 调度与等待原因](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)。本阶段只理解这些调查方向，不展开底层指令研究。

先确认 Event/Duration 的快慢，再找主要等待原因，接着看 scheduler 是否真的缺少发指令的机会，最后用相关访存、吞吐、occupancy 指标和源码交叉解释。本轮 long scoreboard 为 6.16/6.57/8.24 cycles/instruction，在主要等待项中最大；Issue Active 约为 24%/24%/22%，说明值得调查未被隐藏的依赖等待。源码中 max、指数和与输出阶段都读取 input，后续 fmaxf/expf 依赖所读值，这是一个合理调查方向，但当前汇总没有定位哪次读取贡献最大，也不能据此宣布唯一瓶颈。

这些比值不是 kernel 总耗时百分比，不应相加成耗时分解；高 stall 也不自动等于优化它就一定加速。先区分“GPU 在等数据”与“DRAM 带宽已经用满”，不必立即追加复杂实验。

## 两个理解检查的回答与 Review

1. 学习者：“其他 warp 可以执行来隐藏延迟；等待 warp 计入 occupancy，所以等待会使 occupancy 下降。”Review：前半正确，最后的因果关系错误。等待不会把 warp 从驻留集合中移除；若一个 SM 驻留 16 个 warp，上限 32 个，即使其中 12 个在等待、4 个就绪，占用率仍是 `16/32=50%`，不是 `4/32`。等待会减少就绪（eligible）warp，可能降低发出指令的机会，而不是直接降低 occupancy。反过来，驻留 warp 较少可能使延迟更难隐藏，但二者不能混同。
2. 学习者：“不太清楚，是因为同步吗？”导师讲解：同步是尚不能排除的一种成本，但关键不是列出一种其他开销，而是区分证据与结论。本轮较大的 long scoreboard 提示访存结果依赖等待，约 35% 的 DRAM 吞吐没有显示带宽饱和，较低的 Issue Active 提示调查延迟隐藏；这些支持优先调查访存依赖等待，而没有定位哪个 load、哪个缓存层或哪个阶段限制整体性能。barrier、固定延迟计算依赖等仍存在，而且 stall ratio 不是总耗时占比，不能通过排名证明唯一瓶颈。

后续提问：“等待 warp 还占用资源，如何调度其他 warp 隐藏延迟？”导师讲解：寄存器、执行状态与 block 的 shared 分配保留，不代表持续独占算术执行单元。scheduler 从已经驻留且就绪的 warp 中选择下一条指令，不需要先释放等待 warp 的资源。用其他 warp 的独立工作覆盖等待，不是缩短读取本身；若所有候选 warp 都不就绪，仍可能空闲。示意时间线已记录在 [复习问答](../CUDA_复习与面试问答.md#11-nsight-compute-单-kernel-分析)，不将此次提问标作已完成独立分析。

下一步：学习者修正第 1 条，并用自己的话复述第 2 条的证据链；无需新增采集或修改 kernel。可用“16 个驻留 warp、12 个等待、4 个就绪”检查是否区分了 occupancy 与就绪程度。

Review：理解检查已收到初步回答并完成讲解；等待学习者修正 occupancy 因果关系、形成自己的有限瓶颈解释。Day 11 暂不标记验收通过，最终独立 Softmax 验收仍未进行。
