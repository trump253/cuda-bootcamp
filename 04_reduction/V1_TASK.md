# V1：Shared Memory Tree Reduction

## 本阶段目标

把 V0 的单线程串行累加改成 block 内并行的 shared-memory tree reduction。
每个 block 输出一个 partial sum，host 使用同一个 kernel 继续归约 partial sums，
直到 GPU 上只剩一个最终标量。

本版本只学习：

- shared memory 暂存 block 内数据；
- `__syncthreads()` 保证每轮依赖；
- 交错寻址 tree reduction；
- 多轮 kernel 如何把多个 block partial sums 归约成一个标量。

本版本暂不使用 `atomicAdd`、warp shuffle、循环展开或一次加载两个元素。

## 数据流

以 `block_size=256` 为例：

```text
N 个 input
  ↓ 第一轮：ceil(N / 256) 个 block
ceil(N / 256) 个 partial sum
  ↓ 第二轮
更少的 partial sum
  ↓ 重复
1 个 GPU 标量
```

对于 `N=1000`：

```text
第一轮：1000 → 4
第二轮：   4 → 1
passes = 2
```

## 你需要完成

只实现 `v1.cu` 中的 `reduce_sum_v1_kernel`：

1. 每个线程读取至多一个 global-memory 元素并写入 `shared[tid]`。
2. 越界线程向 shared memory 写入加法单位元 `0`，不能提前 return。
3. 完成初始 shared-memory 写入后进行一次 block 级同步。
4. 使用从 1 开始、每轮翻倍的 stride 完成交错寻址 tree reduction。
5. 每一轮加法结束后，所有线程都必须到达同步点。
6. 最终由 thread 0 把 `shared[0]` 写入当前 block 的输出位置。

CPU Reference、输入初始化、ping-pong 临时缓冲区、多轮 kernel launch、正确性
验证和 CUDA Event Benchmark 已全部预置，不需要重写。

## 为什么越界线程不能提前 Return

同一个 block 中的部分线程可能越界，但所有线程随后都要执行
`__syncthreads()`。如果越界线程提前返回，而其他线程继续到达 barrier，会违反
block 级同步要求。正确做法是让越界线程把 `0` 写入 shared memory，然后继续
参与所有同步。

## 验收标准

- 所有正确性尺寸输出 `PASS`。
- 程序退出码为 0。
- Compute Sanitizer 报告 `0 errors` 和 `0 bytes leaked`。
- 连续运行三轮 Benchmark，所有 latency 和 bandwidth 数值显式携带单位。
- 能手工写出 `N=1000, block=256` 时每一轮的输入数、block 数和输出数。
- 能解释为什么 tree reduction 的每轮都需要 `__syncthreads()`。
- 能解释交错寻址为什么会造成 warp divergence，以及下一版准备如何降低它。

## 完成后提交

请提供：

1. `reduce_sum_v1_kernel` 实现。
2. 正确性输出、退出码和 Compute Sanitizer 输出。
3. 暖机后连续三轮 Benchmark。
4. 对同步、多轮 reduction 和 divergence 的解释。

V1 验收后进入 V2：减少 divergence 的 tree reduction。

## V1 验收结果

全部正确性尺寸的绝对误差均为 0；Compute Sanitizer 报告 `0 errors`、
`0 bytes leaked`。暖机后三轮平均结果如下：

| N | 平均 latency | 平均有效带宽 | 相对 V0 speedup |
|---:|---:|---:|---:|
| 1,024 | 0.006895 ms | 0.594 GB/s | 3.04× |
| 16,384 | 0.007022 ms | 9.33 GB/s | 44.24× |
| 262,144 | 0.019320 ms | 54.27 GB/s | 256.08× |
| 1,048,576 | 0.048428 ms | 86.61 GB/s | 408.47× |
| 16,777,216 | 0.639414 ms | 104.95 GB/s | 无 V0 对照 |

`N=1024` 需要两轮 kernel，固定 launch 开销占比较高；数据规模增大后，并行性
逐渐展开。V1 已远快于单线程 V0，但交错寻址会造成严重 warp divergence，且每轮
都包含 block barrier，因此仍远低于硬件可达到的显存带宽。

V1 正确性、Benchmark、Compute Sanitizer 和概念验收完成。下一版本使用连续的
活跃线程区间来减少 divergence。
