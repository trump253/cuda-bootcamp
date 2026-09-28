# Day 2：二维索引与矩阵加法

## 状态

已完成正确性、Compute Sanitizer、CUDA Event Benchmark 和三轮重复性验证。

## 问题定义

输入是两个形状均为 `[rows, cols]` 的 row-major FP32 矩阵：

```text
C[row, col] = A[row, col] + B[row, col]
```

一个 CUDA thread 负责一个输出元素，使用二维 block 和二维 grid。

## 需要自行完成

在 `main.cu` 中完成：

1. `matrix_add_kernel` 的二维线程映射及两个维度的边界检查。
2. `matrix_add_cpu` 参考实现。
3. `max_abs_error`。
4. `run_correctness_case` 中的输入、内存、复制、启动、同步、验证和释放流程。
5. `main` 中的多组测试汇总和正确退出码。

初始 block 固定为：

```text
(16, 16)
```

必须测试：

```text
1 × 1
31 × 33
256 × 256
1000 × 513
```

## 正确性验收标准

- 最大绝对误差不超过 `1e-6`。
- CPU 参考结果与 GPU 输出使用不同缓冲区。
- 行、列方向都能正确处理非整除边界。
- 检查每个 CUDA Runtime API。
- 分别检查 kernel 启动错误和同步执行错误。
- Compute Sanitizer 报告 `0 errors` 和 `0 bytes leaked`。
- 能独立解释 `1000 × 513`、`block=(16,16)` 时的二维 grid 及尾部线程。

## CUDA Event Benchmark

在 `main.cu` 中完成：

1. `benchmark_matrix_add`：warm-up、CUDA Event、多次 kernel 和平均延迟。
2. `run_benchmark_case`：计时区外的数据准备、显存管理、带宽计算和输出。
3. `main` 中全部 benchmark 的有效性汇总。

固定参数：

```text
block      = (16, 16)
warm-up    = 10 次
iterations = 100 次
```

测试规模：

```text
256 × 256
1000 × 513
1024 × 1024
4096 × 4096
```

每个元素产生两次 FP32 读取和一次 FP32 写入：

```text
bytes_per_kernel = rows × cols × 3 × sizeof(float)
effective_bandwidth_GBps = bytes_per_kernel / (latency_ms × 1e6)
```

输出至少包含：

```text
shape
block
warmup
iterations
latency_ms
effective_bandwidth_GBps
```

内存分配、H2D/D2H 复制和 CPU Reference 均不得进入 Event 计时区间。正式计时
循环中不得调用 `cudaDeviceSynchronize()`。

## 实验结果

三轮平均结果：

| Shape | 平均 latency | 平均有效带宽 |
|---|---:|---:|
| 256 × 256 | 0.002502 ms | 314.39 GB/s |
| 1000 × 513 | 0.011823 ms | 520.70 GB/s |
| 1024 × 1024 | 0.024997 ms | 503.38 GB/s |
| 4096 × 4096 | 0.362219 ms | 555.82 GB/s |

较大矩阵稳定在约 500–556 GB/s。`4096 × 4096` 三轮 latency 极差约为
0.022%；小矩阵的固定启动开销占比更高，因此有效带宽更低且波动更明显。

## 计时结论

- CPU wall-clock 表示主机计时器观察到的经过时间，不表示 kernel 在 CPU 上执行。
- kernel launch 是异步的；若不同步，普通主机计时通常只测到 launch 提交时间。
- 若在主机计时中加入同步，结果还会包含 Runtime API 和主机等待开销。
- CUDA Event 在 GPU stream 的执行序列中记录时间，更适合测量 kernel latency。
- `cudaMalloc` 和 H2D/D2H 属于独立操作。计入它们得到的是端到端延迟，而不是
  kernel-only latency；两种指标都有效，但必须明确区分。

## 查看理论显存带宽

先确认实际 GPU 型号：

```bash
nvidia-smi --query-gpu=name,clocks.max.memory --format=csv,noheader
nvidia-smi -q
```

最可靠的方法是根据完整 GPU 型号查厂商官方规格中的 `Memory Bandwidth`。如果规格
只给出显存有效数据率和总线宽度，则：

```text
理论带宽 GB/s = 有效数据率 Gbit/s × 总线宽度 bit ÷ 8
```

不要直接把 `nvidia-smi` 显示的 memory clock 当作有效数据率；GDDR、GDDR6X、
HBM 的时钟报告方式和传输倍率不同。实测有效带宽除以官方理论带宽，可得到带宽
利用率。

本次实际设备为两张 NVIDIA GeForce RTX 2080 Ti。每张卡的 GDDR6 显存时钟为
7000 MHz，DDR 有效数据率为 14 Gbit/s，总线宽度为 352 bit：

```text
单卡理论带宽 = 14 Gbit/s × 352 bit ÷ 8 = 616 GB/s
单卡实测利用率 = 555.82 GB/s ÷ 616 GB/s ≈ 90.2%
```

当前 Matrix Add 只在默认的一张 GPU 上运行，因此应与单卡 616 GB/s 比较。只有
显式编写多 GPU 工作负载并让两张卡同时处理数据时，才讨论聚合带宽。
