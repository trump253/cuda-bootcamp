# Day 12：CUDA Bootcamp 综合复盘

状态：10 个回答已提交，首轮 Review 完成；第 4–7 题待补充，暂未通过综合复盘。Day 11 基础验收已通过，最终独立 Softmax 验收尚未开始。本任务对应学习计划第 12 节的复盘，不新增 kernel，也不添加新的 CUDA 主题。

## 今天的任务与验收

先用自己的话回答下面 10 个已经遇到的问题，每题约 2–4 句话；涉及实现或性能判断时，结合至少一个已有练习或实验说明。可查看自己的源码与历史数据，再用 [复习问答](../CUDA_复习与面试问答.md) 查漏补缺，不直接复制整段答案。

不确定的地方可以明确写出，导师只针对实际缺口补讲。尤其注意区分性能观测与原因推断，不能把最大 stall、低 DRAM 吞吐或高 occupancy 单独当作瓶颈证明。

完成后提交这个文档或直接在对话中回复。验收看概念、例子和证据边界，不要求背长指标名；涉及数值时保留 ms、µs、GB/s、GFLOP/s、% 等对应单位。导师 Review 后再准备从空源文件开始的最终独立 Softmax 任务，不复制旧 kernel。

## 1. CUDA execution model 是什么？

回答：单指令多线程，以warp为执行单位，一条指令控制一个warp 32个线程同时执行，达到并行的目的。

## 2. global、shared、register 分别解决什么问题？

回答：global是GPU容量最丰富的内存，大规模的数据都会保存在全局内存中，特点是数量大，但是访问慢，它的访问是GPU中非常常见的瓶颈；register是单个线程可见的寄存器，速度最快，适合临时变量或者中间值的存储，但是容量有限；shared处于前两者中间，容量比register要大，但是速度不如寄存器，适合做tile切分做局部缓存加速，是gpu最常用来优化访存的部分。

## 3. coalescing 为什么重要？

回答：合并访存可以减少sector事务，同一个warp一起访存如果可以合并，理想状态下只需要4个sector即可搬运完成数据，如果无法合并相同的指令会产生更多的sector，global memory是最慢的，多个sector会极大拖慢速度。

## 4. warp reduction 是怎么工作的？

回答：同一个warp内各个线程寄存器的变量可以通过广播的方式传递给其他线程，从而实现数据互访的效果，减少对共享内存的访问。

## 5. 什么是 memory-bound？

回答：内存瓶颈，运行时间在消耗在了数据搬运上面，DRAM吞吐较高，计算负载较低，在等待数据搬运。典型例子为decode阶段。

## 6. 什么是 compute-bound？

回答：计算瓶颈，运行时间消耗在了计算上面，SM吞吐较高，搬运数据要参与很多计算，搬运带宽跑不满。典型场景为Prefill阶段。

## 7. 如何 benchmark CUDA kernel？

回答：使用cuda event测量kernel时间，或者使用nsys和ncu辅助测试。

## 8. Nsight Systems 和 Nsight Compute 分别解决什么问题？

回答：Nsight Systems以时间线的形式展示整个程序执行过程各个阶段的耗时，包括初始化、数据搬运、运行时间等数据，是宏观视角，可以用来定位数据搬运、流水线并行等问题；Nsight Compute是分析某个核函数的执行数据，包括本次核函数执行的全局内存访问、共享内存访问、bank冲突等数据，是更细节的具体核函数执行的分析。

## 9. 为什么不能只看 occupancy？

回答：occupancy占用率是SM上的warp占SM最大warp的比例，有些warp可能在SM上但是由于等待数据等场景导致空闲，但是依然在SM上占用寄存器等资源。

## 10. 什么时候应该开始 CUDALM？

回答：还缺少一些实际算子，例如RMSNorm，融合算子的开发经验，以及Softmax的独立实现和优化。

## Review 记录

- 学习者提交：10 个回答已完成；以上保留原始答案，以下是导师反馈，不视为学习者已经自行修正。
- 首轮 Review：2026-10-08。第 1、8、9 题方向正确；第 2、3、10 题补充表述边界；第 4–7 题需要学习者复述实际过程与判断依据。
- 当前结论：综合复盘待补充，不进入最终验收、不新增 kernel 或实验。

### 逐题反馈

| 题号 | 导师反馈 |
| --- | --- |
| 1 | SIMT 与 warp 方向正确。补齐层次：grid 由 block 构成，block 的线程分成 warp，在 SM 上调度；一条 warp 指令作用于当时活跃的 lane，不保证 32 个 lane 始终都在做有效工作。 |
| 2 | global 保存输入输出、register 保存线程私有中间值的理解正确。shared 是 block 内线程共享的显式缓存与通信空间，跨线程读写需要适当同步；不能脱离统计范围断言 shared 容量一定大于 register。GEMM 的 A/B tile 是已有例子。 |
| 3 | 减少无效 sector 的方向正确，但 4 个 sector 有条件：32 个活跃 lane 各读 1 个 FP32、地址连续且起点按 32 字节对齐。当前 float4 每 lane 读 16 字节，连续访问则为 16 个 sector，仍是合并访问。L1/TEX sector 也不直接等于 DRAM 流量，速度收益仍需测量。 |
| 4 | 说明了 shuffle 减少 shared 交换，但没有说明怎样规约。广播是多个 lane 读取同一个来源；当前 `__shfl_down_sync` 让各 lane 读取自己的 `lane + delta`，再做加法或 max。delta 为 16、8、4、2、1，最终由 lane 0 使用完整 warp 结果；它不能跨 warp。 |
| 5 | memory-bound 指性能主要受内存子系统限制，既可能受带宽限制，也可能受访存延迟及其隐藏能力限制。因此 DRAM 吞吐较高不是必要条件。Day 11 的 DRAM 约 35% 与 long scoreboard，只支持优先调查访存依赖，尚不能证明唯一瓶颈。 |
| 6 | compute-bound 指性能主要受计算执行能力限制，而不只是“做了很多计算”。SM Throughput 是汇总指标，不等于 FP32 算术管线利用率或 GFLOP/s；高 SM、低 DRAM 不能单独证明计算瓶颈。decode/prefill 只能作为有条件的常见例子，不能替代当前工作负载的测量。 |
| 7 | 只列工具名还不够，需要说明可复现流程：先正确性与错误检查，分配/复制置于计时外，暖机，同一 stream 记录 start → 重复 launch → stop，同步 stop 后取 elapsed/iterations。固定 GPU、shape、编译配置，至少三轮，报告 ms 与波动；ncu/Sanitizer 下的时间不能直接混入普通 Event 基线。 |
| 8 | 已正确区分系统时间线与单 kernel 内部分析。 |
| 9 | 已正确解释等待 warp 仍驻留、仍占资源；可再强调 occupancy 不等于就绪 warp 比例或指令发出比例。 |
| 10 | 最终独立 Softmax 闭环确实尚缺。按计划，RMSNorm/fusion 属于进度快时的可选补缺，不是额外毕业条件；最终验收通过后进入 CUDALM，不以“所有算子都学过”为前提。 |

访存计数口径见 [NVIDIA L1/TEX 表说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables)，shuffle 语义见 [CUDA 11.8 Warp Shuffle](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#warp-shuffle-functions)，计时步骤见 [CUDA 11.8 GPU Timers](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers)。瓶颈判断结合 [Nsight Compute 调度与吞吐说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) 与本仓库 Day 11 的既有结果，不把汇总指标当作因果证明。

### 只需补充的四个回答

可直接在对话中提交，或在本小节追加“修订回答”，不必重写全部十题，也不用新采集：

1. 第 4 题：结合自己的 Softmax V2，说清五轮 shuffle 如何合并值、哪个 lane 得到结果，以及 8 个 warp 的结果如何合并。
2. 第 5 题：用 Day 11 的既有结果说明“DRAM 未跑满”为什么不能排除访存限制，当前究竟证明了什么、没证明什么。
3. 第 6 题：用已有 GEMM 练习说明算术强度/复用与 compute-bound 判断的关系，以及为什么高 SM Throughput 不足以定论。
4. 第 7 题：不写完整代码，用顺序步骤描述你已经使用过的 Event Benchmark；说明同步对象、平均值算法和计时范围。

以上实际缺口确认后再准备从空源文件开始的最终独立 Softmax 验收框架；最终 kernel、优化与实验仍由学习者完成。
