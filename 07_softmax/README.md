# Day 9：Softmax V0–V3

本节输入、输出均为行主序 FP32 矩阵，形状 `[rows, hidden]`。每一行独立执行
Softmax。按计划依次完成 V0 逐行朴素版、V1 Shared Memory、V2 Warp Shuffle、
V3 合理向量化；每版都要有正确性、Benchmark 与必要的 Profile 证据，不能只凭
源码推断加速。V0/V1 已验收；当前开放 V2，V3 暂不提前实现。

## 当前文件

- `v0.cu`：学习者已实现 V0 逐行串行 GPU kernel；目前保留了原模板的 TODO 注释。
- `v1.cu`：学习者已实现每行一个 block 的 Shared Memory max/sum 规约。
- `v2.cu`：复用 V1 的局部扫描与写回，warp/block 规约保留中文 TODO。
- `softmax_harness.h`：共用的确定性输入、CPU Reference、GPU 正确性校验、
  CUDA Event Benchmark 和单次 kernel Profile 入口；不用重新写测试样板。
- `DAY9_V0_TASK.md`：已完成的 V0 任务说明。
- `DAY9_V1_TASK.md`：已完成的 V1 任务、验收标准和提交内容。
- `DAY9_V2_TASK.md`：当前 V2 Warp Shuffle 任务与验收标准。

V1 的初版与列映射修正版均已完成正确性、Benchmark/Profile 对照。

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
V1 与该基线采用相同 shape、GPU 和构建条件对照。

## V1 初版：正确，但仍是跨 sector 访存

学习者实现的 V1 使用局部 max、shared-memory tree max、局部指数和、
shared-memory tree sum，再归一化写回。所有 17 个 shape 正确性 `PASS`，
程序退出码为 0；Compute Sanitizer memcheck 报告 `0 errors`、
`0 bytes leaked`，退出码为 0。无元素线程分别贡献 `-∞` 和 `0`，
各轮 barrier 均由整个 block 到达；初版代码未发现明显的同步位置错误。

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
V0 对应为 `49152` request、`1572864` sector。初版 V1 在 `hidden=4096` 时，
`cols_per_thread=16`，同一轮读取的列为 `tid×16+i`，相邻 lane 相隔
`16×4=64` 字节，各占不同 sector。V1 的 request 数可按
`128 block×8 warp/block×16 轮×3 遍=49152` 核对。

因此初版测得的加速与“更多 block 并行、每线程串行工作更少”相符，
但这些数据不能分离各项因素的贡献。当时待验证的问题是：保持 block
配置与算法不变，只改变列分配，让同一轮的相邻 lane 处理相邻列，
`sector/request` 和实际延迟会怎样变化？

## V1 列映射修正版：正确性、Benchmark、Profile

学习者将 max、指数和、写回三遍循环统一改为从 `tid` 起步、每轮跨
`blockDim.x=256` 列。固定循环轮次时，warp 内相邻 lane 访问相邻 FP32
元素；下一轮再由同一线程处理相距 256 列的元素。17 个正确性用例全部
`PASS`，程序退出码为 0；Compute Sanitizer memcheck 为 `0 errors`、
`0 bytes leaked`，退出码为 0。

同一 GPU、同一构建配置下，V0/V1 整程序预热后交替运行三轮。下表是
CUDA Event 的 kernel-only 延迟，warm-up 10 次、正式迭代 100 次：

| 形状 | V0 三轮 latency（ms） | 修正版 V1 三轮 latency（ms） | 修正版 V1 平均（ms） |
| --- | --- | --- | ---: |
| `(1,128)` | 0.015221 / 0.010771 / 0.010772 | 0.002377 / 0.002378 / 0.002389 | 0.002381 |
| `(16,512)` | 0.076347 / 0.053679 / 0.053453 | 0.002751 / 0.002765 / 0.002748 | 0.002755 |
| `(128,1024)` | 0.440791 / 0.313692 / 0.312934 | 0.003488 / 0.003484 / 0.003503 | 0.003492 |
| `(128,4096)` | 1.714258 / 1.252110 / 1.252638 | 0.005181 / 0.005172 / 0.005180 | 0.005178 |

V0 本次每个尺寸的第一轮都明显偏高，不能直接取这三轮平均值作精确
speedup 基线；后两轮与前述 V0 稳定基线相近。针对相同的
`(128,4096)`，V1 初版三轮平均 `0.025122 ms`，修正版平均
`0.005178 ms`，同配置下约快 `4.852×`，延迟降低约 `79.39%`。

学习者对修正版采集的一次 Nsight Compute Profile：
`gpu__time_duration.sum=10.94 µs`，L1/TEX global-load request 为
`49152`、sector 为 `196608`，即 `4 sector/request`。与 V1 初版
`49152 request`、`1572864 sector`、`32 sector/request` 相比，请求数
不变而 sector 数降低 `87.5%`（8 倍）。这与一个 warp 连续读取
32 个 FP32 元素覆盖 4 个对齐 32 字节 sector 相符，是本次读取侧
coalescing 改善的直接证据。L1/TEX sector 不等于实际 DRAM 传输量。

本次列映射还改变了写回的地址分布，并可能改变编译后指令路径；不能把
全部 `4.852×` 加速仅归因于 global load sector 减少。Profile 条件下的
`10.94 µs` 也不应与正常 Event 的 `0.005178 ms` 混作同一计时口径。
至此 V1 的 Reference、Correctness、Benchmark、Profile、Optimization、
Re-benchmark 和 Notes 已闭环；下一版按计划由学习者实现 V2 Warp Shuffle。

## V1 补充实验：把指数结果暂存到 output

学习者又尝试在求指数和的循环里计算 `e=expf(x-row_max)`，先写入
`output`，归一化时从 `output` 读回并乘 `inv_sum`。这在源码层面将每个
元素的 `expf` 从两次降到一次，但也把一次最终写回变为“中间写回 +
读回 + 最终写回”。17 个正确性用例全部 `PASS`；学习者随后表示已补跑
Compute Sanitizer，但尚未提供退出码及摘要原始输出，因此内存安全复验
仍待证据确认。

先前已验收的 V1 列映射修正版与本次暂存变体的 CUDA Event 三轮数据：

| 形状 | 直接重算 `expf`：三轮 / 平均（ms） | 暂存到 output：三轮 / 平均（ms） | 暂存版延迟变化 |
| --- | --- | --- | ---: |
| `(1,128)` | 0.002377 / 0.002378 / 0.002389；0.002381 | 0.003355 / 0.003381 / 0.003360；0.003365 | 增加约 41.3% |
| `(16,512)` | 0.002751 / 0.002765 / 0.002748；0.002755 | 0.003809 / 0.003828 / 0.003802；0.003813 | 增加约 38.4% |
| `(128,1024)` | 0.003488 / 0.003484 / 0.003503；0.003492 | 0.005013 / 0.005013 / 0.005008；0.005011 | 增加约 43.5% |
| `(128,4096)` | 0.005181 / 0.005172 / 0.005180；0.005178 | 0.009075 / 0.009079 / 0.009122；0.009092 | 增加约 75.6% |

这两批数据不是同一轮交替 A/B 运行，因此百分比是已观察到的对照，
不是精确的硬件代价分解；但每组内部的三轮值都较稳定，且四种形状
均显示回退。暂存版的一次 ncu Profile 中，`(128,4096)` 的 global-load
request/sector 仍为 `49152/196608`，即 `4 sector/request`；
Profiler 时长为 `12.16 µs`，而此前直接重算版为 `10.94 µs`。
正常 Event 时间仍是主要性能判据，不与 ncu 的绝对时长混算。

从源码看，直接重算版每元素是 3 次 global load、1 次 global store、
2 次 `expf`；暂存版是 3 次 global load、2 次 global store、1 次
`expf`。第三遍从读取 input 变成读取 output，load 次数没有减少，
另多一次 output store。global-load Profile 数值不变与此相符；
store 的实际指令和缓存/DRAM 流量尚未采集，不能断言单一的回退原因。
原修正版继续作为 V1 性能基线；学习者已将工作树中的 `v1.cu` 恢复为
归一化时重新计算 `expf` 的版本。V2 用它做同配置对照，暂存实验只保留
为一次负收益记录。

## V2 Warp Shuffle：正确性、Benchmark、Profile

学习者独立实现 warp max/sum 和两次跨 warp 规约。17 个 shape 全部
`PASS`、程序退出码为 0；Compute Sanitizer memcheck 为 `0 errors`、
`0 bytes leaked`，退出码为 0。第一处 block barrier 保证所有 warp 的
partial 已写入 shared；第二处保证 warp 0 写出的行 max/sum 已可供
整个 block 读取。warp 0 中只有前 8 个 lane 读取有效 partial，
其余 lane 分别用 `-INFINITY`、`0` 参加 full-mask 规约。

同一 GPU、同一构建配置下，V1/V2 整程序预热后交替三轮；下表为
CUDA Event kernel-only latency（warm-up 10、正式迭代 100）：

| 形状 | V1 三轮 / 平均（ms） | V2 三轮 / 平均（ms） | V1/V2 平均延迟比 |
| --- | --- | --- | ---: |
| `(1,128)` | 0.003342 / 0.003340 / 0.003388；0.003357 | 0.003052 / 0.003277 / 0.002867；0.003065 | 1.095× |
| `(16,512)` | 0.003902 / 0.003891 / 0.003850；0.003881 | 0.003396 / 0.003501 / 0.003393；0.003430 | 1.131× |
| `(128,1024)` | 0.004964 / 0.004936 / 0.004918；0.004939 | 0.004112 / 0.004096 / 0.004235；0.004148 | 1.191× |
| `(128,4096)` | 0.007373 / 0.007368 / 0.007351；0.007364 | 0.006737 / 0.006713 / 0.006717；0.006722 | 1.095× |

四个形状均显示 V2 更快；例如 `(128,4096)` 延迟降低约 `8.71%`。
本轮 V1 绝对延迟不同于先前的 `0.005178 ms`，不能跨会话混算速度比，
以同轮交替测得的 V1/V2 数据为准。小形状结果仍受短 kernel 波动影响。

学习者对 `(128,4096)` 各采集一次 Nsight Compute：

| 指标 | V1 | V2 |
| --- | ---: | ---: |
| `gpu__time_duration.sum` | 11.26 µs | 10.40 µs |
| `smsp__inst_executed_op_shared_ld.sum` | 34816 inst | 2304 inst |
| `smsp__inst_executed_op_shared_st.sum` | 18432 inst | 2304 inst |
| `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | 8.02% | 6.97% |

shared load/store 的 warp 指令计数分别降低约 `93.38%`/`87.50%`，
barrier stall 比例下降 `1.05` 个百分点；这些与 Event 加速方向一致，
但 stall 百分比不是耗时占比，不能单独证明唯一的加速原因。ncu 的
`11.26/10.40 µs` 也不与正常 Event 的绝对时长混算。

### 本次 shared 指令计数如何核对

Profile 使用 128 个 block、每 block 256 线程即 8 个 warp，分别做
max/sum 两次规约。计数单位是执行过的 **warp 级 SASS 指令**，不是
逐线程读写次数，也不是 shared-memory 请求字节数；相关口径见
[NVIDIA Nsight Compute 指标说明](https://docs.nvidia.com/nsight-compute/NsightCompute/)。

- V1 每次规约：初始化 shared 有 `8` 条 warp store；8 轮 tree 各有
  `2` 条 shared load、`1` 条 shared store，当前编译器把条件访问
  生成为带谓词的指令，每轮都由 8 个 warp 执行；最终各 warp 读取
  行标量，共 `8` 条 load。因此两次规约的 load 为
  `128 block × 2 × 8 warp × (8 轮 × 2 load + 1 final load)
  = 34816 inst`；store 为
  `128 × 2 × 8 × (1 init store + 8 轮 × 1 store)=18432 inst`。
  部分 warp 的条件为假，仍可计入此处的指令数，不代表都产生有效
  shared-memory 请求。
- V2 每次规约：8 个 warp 各写 1 个 partial，再由 warp 0 写 1 个
  行标量，所以每 block 有 `9` 条 warp store；warp 0 读取 8 个
  partial 对应 `1` 条 warp load，随后 8 个 warp 各读取行标量一次，
  共 `9` 条 warp load。两次规约、128 个 block 给出
  `128 × 2 × 9 = 2304 inst`，与 load/store 两项实测均一致。

上述公式依赖当前编译结果；改源码、编译参数或架构后，不能保证
仍是相同指令数。若要区别“warp 发出了带谓词的指令”与“至少一个
lane 实际参与”，可继续采集同名指标的 `_pred_off_all` 或
`_pred_on_any` 变体。V2 的正确性、Benchmark、Profile、Notes
闭环已完成；Day 9 的 V3 尚未开始。
