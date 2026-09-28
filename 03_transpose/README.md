# Day 3–4：Matrix Transpose

## 完成状态

本节已完成 V0 Naive、V1 Shared Memory Tiled、padding 优化、CUDA Event
Benchmark 与 Nsight Compute bank-conflict 分析，所有版本均通过正确性与
Compute Sanitizer 验证。

## 问题定义

输入是 row-major FP32 矩阵：

```text
input  shape = [rows, cols]
output shape = [cols, rows]
output[col, row] = input[row, col]
```

一个 CUDA thread 负责转置一个元素，初始 block 为 `(16,16)`。

## 需要自行完成

在 `main.cu` 中完成：

1. `transpose_naive_kernel` 的二维索引、边界和输出地址。
2. `transpose_cpu` 参考实现。
3. `max_abs_error`。
4. `run_correctness_case` 的内存、复制、启动、验证与释放流程。
5. `main` 中全部测试的汇总和正确退出码。

必须测试：

```text
1 × 1
31 × 33
32 × 33
256 × 256
1000 × 513
```

## 验收标准

- 所有测试最大绝对误差不超过 `1e-6`。
- 输出缓冲区按 `[cols, rows]` 的 row-major 布局解释。
- 行、列方向均正确处理非整除边界。
- 所有 CUDA API、launch error 和异步执行错误均被检查。
- 程序退出码为 0。
- Compute Sanitizer 报告 `0 errors` 和 `0 bytes leaked`。
- 能解释朴素映射中读取与写入分别是否连续，以及哪一侧可能造成非合并访存。

## Benchmark 方法

完成 `benchmark_transpose_naive` 和 `run_benchmark_case`：

```text
block      = (16,16)
warm-up    = 10
iterations = 100
```

测试：

```text
256 × 256
1000 × 513
1024 × 1024
4096 × 4096
```

Transpose 每个元素读取一次 FP32、写入一次 FP32：

```text
bytes_per_kernel = rows × cols × 2 × sizeof(float)
effective_bandwidth_GBps = bytes_per_kernel / (latency_ms × 1e6)
```

有效带宽是算法字节数除以 kernel latency，用于同一算法不同版本之间的比较；
它不等于 Nsight Compute 测得的真实 DRAM throughput。

## V0 Baseline

三轮平均结果：

| Shape | 平均 latency | 平均有效带宽 |
|---|---:|---:|
| 256 × 256 | 0.003015 ms | 173.87 GB/s |
| 1000 × 513 | 0.008372 ms | 490.17 GB/s |
| 1024 × 1024 | 0.023268 ms | 360.53 GB/s |
| 4096 × 4096 | 0.324170 ms | 414.03 GB/s |

`4096 × 4096` 达到 RTX 2080 Ti 单卡理论带宽的约 67.2%，低于 Matrix Add 的
约 555.82 GB/s。V0 的 input 读取较易合并，而 output 写入以 `rows` 为步长，
需要更多分散的内存事务。

该有效带宽使用算法字节数计算，不等于真实 DRAM throughput。重复访问同一缓冲区
可能命中 L2 cache，尤其是较小尺寸；真实 DRAM 指标留到 Nsight Compute 阶段验证。

## V1：Shared Memory Tiled，无 Padding

V1 使用 `16 × 16` shared-memory tile，将 V0 的跨步 global write 改成合并
global write。三轮平均结果如下：

| Shape | 平均 latency | 平均有效带宽 | 相对 V0 speedup |
|---|---:|---:|---:|
| 256 × 256 | 0.002763 ms | 189.80 GB/s | 1.091× |
| 1000 × 513 | 0.008465 ms | 484.86 GB/s | 0.989× |
| 1024 × 1024 | 0.020058 ms | 418.22 GB/s | 1.160× |
| 4096 × 4096 | 0.280313 ms | 478.81 GB/s | 1.156× |

V1 已实现合并 global load/store，但转置读取 `tile[threadIdx.x][threadIdx.y]`
在 stride=16 时产生 8-way shared-memory bank conflict。

## V2：Shared Memory Tiled，Padding=1

将 shared-memory tile 从 `[16][16]` 改为 `[16][17]`，kernel 的其他逻辑、
测试规模、warm-up 和 iterations 均保持不变。三轮平均结果如下：

| Shape | 平均 latency | 平均有效带宽 | 相对 V0 speedup | 相对无 Padding speedup |
|---|---:|---:|---:|---:|
| 256 × 256 | 0.002717 ms | 193.97 GB/s | 1.110× | 1.017× |
| 1000 × 513 | 0.006960 ms | 589.65 GB/s | 1.203× | 1.216× |
| 1024 × 1024 | 0.018293 ms | 458.57 GB/s | 1.272× | 1.096× |
| 4096 × 4096 | 0.259670 ms | 516.88 GB/s | 1.248× | 1.080× |

以波动较小的 `4096 × 4096` 为主要结论：padding 相对无 padding 使 latency
降低约 7.36%，相对 Naive 使 latency 降低约 19.90%。有效带宽从 V0 的
414.03 GB/s 提高到 516.88 GB/s，约为 RTX 2080 Ti 616 GB/s 理论显存带宽
的 83.9%。该百分比仍是算法有效带宽与理论带宽的比值，不等于 DRAM 利用率。

## Nsight Compute：Bank Conflict 证据

Profile 配置为 `4096 × 4096`、`block=(16,16)`、`grid=(256,256)`：

```text
block 数量       = 256 × 256 = 65536
每 block warp 数 = 256 / 32 = 8
总 warp 数        = 65536 × 8 = 524288
```

| 版本 | Shared load conflict | Shared store conflict | 每 warp 总 conflict |
|---|---:|---:|---:|
| Padding=0 | 3670016 | 0 | 7 |
| Padding=1 | 524288 | 524288 | 2 |

Padding 使 shared load conflict 降低 85.71%，load 与 store 的总 conflict
降低 71.43%。Padding=0 的 8-way load conflict 在计数器中表现为每 warp
7 个额外 conflict；padding=1 时，由于一个 warp 横跨两个 16-thread row，
shared load 和 store 各剩一个 2-way conflict。

## 最终结论

- V0：global load 合并，但转置后的 global store 跨步。
- V1：shared-memory tile 将 global load/store 都变为合并访问，但引入严重的
  转置读取 bank conflict。
- V2：padding 改变 shared-memory 行跨度，显著降低 bank conflict，并由
  CUDA Event 与 Nsight Compute 两类证据共同证明优化成立。
- 小尺寸 kernel 受固定开销和频率波动影响更明显，优化判断优先采用多轮稳定的
  大尺寸结果。
