# Day 5–6：Sum Reduction V0

## 本阶段目标

输入一个 FP32 向量：

```text
x[0], x[1], ..., x[N-1]
```

输出一个标量：

```text
sum = x[0] + x[1] + ... + x[N-1]
```

本任务只完成 V0 串行 CUDA 基线，不使用 shared memory、atomic、warp shuffle
或多 block reduction。它的目的不是获得好性能，而是建立 Reduction 的正确性、
数值误差和 Benchmark 基线。

## 最少理论

Vector Add 中每个输出元素相互独立，可以直接分配给不同线程。Reduction 不同：
多个输入最终合并成同一个输出，线程之间存在数据汇聚关系。后续版本需要解决：

```text
如何让多个线程并行产生局部和？
如何安全地合并这些局部和？
如何在 block 和 warp 内同步？
```

V0 暂时回避协作问题：只让全局线程 0 串行读取并累加全部元素，其余线程不工作。

## V0 Kernel 状态

`reduce_sum_v0_kernel` 已完成以下逻辑：

1. 计算当前线程的全局一维编号。
2. 只有全局线程 0 执行累加，其余线程立即返回。
3. 使用线程私有的 FP32 累加变量遍历 `[0, n)`。
4. 将最终结果写入 `output[0]`。

CPU Reference、输入初始化、设备内存、CUDA 错误检查、正确性验证和 CUDA Event
Benchmark 已经预置，不需要重写。

所有性能输出必须让数值显式携带单位，例如：

```text
latency=19.781560 ms, effective_bandwidth=0.21 GB/s
```

## 测试与验收

正确性尺寸：

```text
1, 31, 32, 33, 255, 256, 257, 1000, 2^20 + 3
```

要求：

- 所有测试输出 `PASS`，绝对误差不超过 `1e-5`。
- 程序退出码为 0。
- Compute Sanitizer 报告 `0 errors` 和 `0 bytes leaked`。
- 连续运行三次 Benchmark，记录 latency 和有效带宽。
- 能解释为什么启动了 256 个线程，却只有一个线程工作。
- 能解释为什么这个版本无法利用 GPU 的并行能力和显存带宽。

Benchmark 尺寸：

```text
2^10, 2^14, 2^18, 2^20
```

V0 为避免串行基线耗时过长，使用：

```text
warm-up = 2
iterations = 10
```

有效带宽按主要输入流量计算：

```text
effective_bandwidth_GBps = N × sizeof(float) / (latency_ms × 1e6)
```

## 完成后提交

请提供：

1. `reduce_sum_v0_kernel` 的实现。
2. 完整正确性输出和退出码。
3. Compute Sanitizer 输出。
4. 三轮正常 Benchmark 输出。
5. 对以下问题的回答：为什么 Reduction 不能像 Vector Add 一样让每个线程
   独立写最终输出？V0 的性能瓶颈是什么？

V0 验收后再进入 V1 Shared Memory Tree Reduction。

## V0 验收结果

V0 已通过全部正确性测试，最大绝对误差为 0；Compute Sanitizer 报告
`0 errors`、`0 bytes leaked`。丢弃一次整程序暖机后，三轮平均结果如下：

| N | 平均 latency | 平均有效带宽 | 三轮 latency 极差占均值 |
|---:|---:|---:|---:|
| 1,024 | 0.020980 ms | 0.195 GB/s | 1.263% |
| 16,384 | 0.310640 ms | 0.211 GB/s | 0.038% |
| 262,144 | 4.947514 ms | 0.212 GB/s | 0.014% |
| 1,048,576 | 19.781326 ms | 0.212 GB/s | 0.008% |

较大规模的有效带宽稳定在约 0.212 GB/s，说明单线程串行基线既无法形成有效的
warp 级合并访问，也没有足够的 warp 隐藏 global-memory latency。V0 只发射一个
block；其中七个 warp 在条件判断后全部退出，剩余 warp 也只有 lane 0 执行循环。
因此只有一个 SM 接收到该 block，其余 SM 在此 kernel 期间没有工作。

V0 正确性、Benchmark 与内存安全验收完成，下一版本为 V1 Shared Memory Tree
Reduction。

## V1–V3 优化结果

后续版本保持相同的多轮 GPU reduction、测试规模和 CUDA Event 计时方式：

- V1：交错寻址的 Shared Memory Tree Reduction。
- V2：连续低编号线程参与，减少取模与交错控制工作。
- V3：warp 内使用 `__shfl_down_sync`，只通过 shared memory 交换 8 个 warp sum。

暖机后三轮平均结果：

| N | V1 latency | V2 latency | V3 latency | V3 有效带宽 |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.006895 ms | 0.005464 ms | 0.004902 ms | 0.84 GB/s |
| 16,384 | 0.007022 ms | 0.005486 ms | 0.004961 ms | 13.21 GB/s |
| 262,144 | 0.019320 ms | 0.014094 ms | 0.009718 ms | 107.90 GB/s |
| 1,048,576 | 0.048428 ms | 0.033143 ms | 0.019637 ms | 213.59 GB/s |
| 16,777,216 | 0.639414 ms | 0.432722 ms | 0.247308 ms | 271.36 GB/s |

在 `N=2^24` 上，V3 相对 V2 加速 1.750×。同尺寸 Nsight Compute 对照如下：

| 指标 | V2 | V3 | 变化 |
| --- | ---: | ---: | ---: |
| Warp instructions | 53,805,056 | 17,956,864 | 减少 66.63% |
| Shared load instructions | 8,454,144 | 65,536 | 减少 99.22% |
| Shared store instructions | 4,718,592 | 524,288 | 减少 88.89% |
| Barrier stall / active warp | 22.41% | 6.39% | 降低 16.02 个百分点 |
| Warps stalled at barrier | 246,446,359 | 31,432,461 | 减少 87.25% |

因此，V3 的加速有 Benchmark 和 Profile 双重证据：它减少了 shared-memory 指令、
block barrier 等待和总体动态指令。不能把收益只归结为“shuffle 单条指令更快”，
也不能根据静态 barrier 次数推导 stall 指标必须严格按相同比例下降。

## 阶段结论

V3 是当前 V0–V3 中最快的版本。小规模 `N=1024` 只有 4 个首轮 block，无法填满
RTX 2080 Ti 的全部 SM，而且两轮 kernel launch 等固定成本在约 5 微秒的总时间中
占比较高，因此 V3 相对 V2 只有约 1.115×。大规模输入产生足够多的 block，GPU
得到充分并行，每个 block 减少的 shared-memory、barrier 和指令开销会在大量 block
上累积，因此 `N=2^24` 的加速扩大到约 1.750×。不能表述成“block 数越多，单个
block 本身就会变快”。

Sum Reduction 的总工作量是 `O(N)`，因为总共仍需约 `N-1` 次加法；tree reduction
的 `O(log N)` 描述的是具有足够并行资源时的关键路径深度。每个输入至少需要从
global memory 读取一次，而每个元素只对应约一次加法，算术强度很低，因此大规模
Reduction 通常更偏 memory-bound。不过当前 V3 有效带宽约 271.36 GB/s，尚未达到
616 GB/s 理论显存带宽，并且 Profile 显示同步与动态指令仍显著影响性能，所以更
准确的描述是：算法具有 memory-bound 倾向，但当前实现并非只受 DRAM 峰值带宽
限制。

Reduction V0–V3 已完成 Reference、Correctness、Benchmark、Profile、Optimization、
Re-benchmark 和 Notes 的完整闭环。
