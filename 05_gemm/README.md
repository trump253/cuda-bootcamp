# Day 7：Naive GEMM 与 Shared Memory Tiled GEMM

## 范围与停止条件

本阶段只完成：

```text
CPU Reference
→ V1 Naive CUDA GEMM
→ Correctness / Benchmark / Profile
→ V2 Shared Memory Tiled GEMM
→ Re-benchmark / cuBLAS 单次对照 / Notes
```

完成 Naive 与 Tiled 两版后停止，不进入 register tiling、Tensor Core、WMMA、
CUTLASS、CuTe 或 async copy。

矩阵均使用 row-major 布局：

```text
A: M × K
B: K × N
C: M × N
C[row, col] = Σ A[row, inner] × B[inner, col]
```

## 文件分工

- `gemm_harness.h`：已提供 CPU Reference、数据初始化、误差验证、CUDA Event、
  Benchmark、GFLOP/s 计算和单 kernel Profile 模式，不需要重写。
- `naive.cu`：V1 基线，一个线程计算一个输出元素。
- `tiled.cu`：V2 Shared Memory Tiled GEMM，已完成正确性、Benchmark 与 Profile。
- `cublas_compare.cu`：已完成 cuBLAS SGEMM 正确性与参考性能对照。
- `NAIVE_TASK.md`：V1 的任务、验收标准和结果。
- `TILED_TASK.md`：V2 的接口、验证与已完成结果。
- `CUBLAS_TASK.md`：cuBLAS 对照的运行命令与验收记录。

## 为什么 GEMM 使用 GFLOP/s？

一个 `M × K` 与 `K × N` 的 GEMM，按常用约定计算：

```text
FLOPs = 2 × M × N × K
GFLOP/s = FLOPs / latency
```

这里把一次乘法和一次加法计为两个浮点操作。所有输出显式带有 `ms` 和
`GFLOP/s` 单位。

## 推进顺序

V1 已完成正确性、Compute Sanitizer、三轮稳定 Benchmark 和最小 Profile。
V2 已完成正确性、Compute Sanitizer、三轮 Benchmark 和同条件 Nsight Compute
对照。学习计划规定的一次 cuBLAS 对照和结果解释也已完成。Day 7 GEMM 验收结束。

## V1 当前结果

Naive GEMM 已通过全部正确性测试，退出码为 0；Compute Sanitizer 报告
`0 errors`、`0 bytes leaked`。暖机后三轮平均值如下：

| Shape | 平均 latency | 平均性能 | latency 极差占均值 |
| --- | ---: | ---: | ---: |
| `256³` | 0.042171 ms | 795.67 GFLOP/s | 0.368% |
| `512³` | 0.305399 ms | 878.97 GFLOP/s | 0.094% |
| `1024³` | 2.348016 ms | 914.59 GFLOP/s | 0.019% |

`M=31, N=33, block=(16,16)` 时，grid 为 `(3,2)`。右下角 block 只覆盖 15 个
有效行和 1 个有效列，因此只有 15 个有效线程，另有 241 个线程空闲。

对固定 inner，同一个 warp 执行同一条 load 时，A 在每个 16-thread 半 warp 中读取
同一个标量，属于广播式复用；B 在每个半 warp 中读取连续列，具有合并访存模式。
随着 inner 改变，单个线程的 A 地址连续、B 地址跨 N，但这不是判断 warp 合并访存
的观察方向。A/B 都会被计算不同输出元素的线程重复使用，这正是下一版 tiled GEMM
要显式搬入 shared memory 的数据复用机会。

## V1 Nsight Compute 基线

学习者在 `M=N=K=1024`、`block=(16,16)`、`grid=(64,64)` 下采集了一次
单 kernel Profile：

| 指标 | V1 Naive |
| --- | ---: |
| GPU duration | 2.35 ms |
| SM throughput / peak sustained | 62.50% |
| DRAM throughput / peak sustained | 0.90% |
| L1/TEX global-load requests | 67,108,864 requests |
| L1/TEX global-load sectors | 134,216,439 sectors |
| 平均 sectors/request | 约 2.00 |

request 数恰好等于 `4096 blocks × 8 warps/block × 1024 K-iterations × 2 operands`，
与当前每个 warp 在 K 循环中分别读取 A/B 的结构吻合。sector 与 request 都是
L1/TEX 路径上的计数，不等于 DRAM 传输字节数；这组指标也不足以单独定位所有
性能瓶颈。当前 `--profile` 模式将 A/B 填 0，与正常 Benchmark 的非零输入不同。
后续 V2 使用同一 Profile 模式作单变量对照。

## V2 Tiled GEMM 当前结果

五组正确性测试全部 PASS，程序退出码为 0；Compute Sanitizer 报告 `0 errors`、
`0 bytes leaked`。暖机后三轮 Benchmark 平均值：

| Shape | V1 latency | V2 latency | V2 平均性能 | V2 相对 V1 加速 |
| --- | ---: | ---: | ---: | ---: |
| `256³` | 0.042171 ms | 0.029995 ms | 1118.69 GFLOP/s | 1.406× |
| `512³` | 0.305399 ms | 0.200406 ms | 1339.46 GFLOP/s | 1.524× |
| `1024³` | 2.348016 ms | 1.505280 ms | 1426.64 GFLOP/s | 1.560× |

V2 的 tile 结构让每个阶段的 A 行数据被同一输出行的不同列线程使用，B 列数据
被同一输出列的不同行线程使用。两次 block barrier 分别保护“加载完才能读取”和
“读完才能覆写”。尾部补 0 保证不完整 tile 对点积的贡献为 0，也避免沿用上一
阶段留在 shared memory 中的值。

## V1/V2 Nsight Compute 对照

学习者在相同 `M=N=K=1024`、`block=(16,16)`、`grid=(64,64)` 和全 0
`--profile` 输入下完成单 kernel 对照：

| 指标 | V1 Naive | V2 Tiled | 变化 |
| --- | ---: | ---: | ---: |
| GPU duration | 2.35 ms | 1.51 ms | 约 1.56× 加速 |
| L1/TEX global-load requests | 67,108,864 | 4,194,304 | 减少 93.75%，即 16× |
| L1/TEX global-load sectors | 134,216,439 | 16,656,832 | 减少约 87.59%，约 8.06× |
| 平均 sectors/request | 约 2.00 | 约 3.97 | 每请求覆盖更多 sector |
| SM throughput / peak sustained | 62.50% | 73.23% | 提高 10.73 个百分点 |
| DRAM throughput / peak sustained | 0.90% | 1.41% | 提高 0.51 个百分点 |

V2 request 数恰好等于 `4096 blocks × 8 warps/block × 64 个 K 阶段 × 2 个操作数`。
同一阶段加载 A/B tile 后，数据在 shared memory 中被多个输出线程复用，使
L1/TEX global-load request 数下降 16 倍。V2 每个 request 平均触及更多 sector，
因此 sector 总数约下降 8.06 倍，而非 16 倍。两组指标都位于 L1/TEX 路径，
不能直接等同于实际 DRAM 字节数。

DRAM throughput 百分比上升与 kernel 变短可以同时发生；它不说明 V2 搬运了更多
DRAM 字节。SM throughput 也不是专门的 FP32 计算单元利用率。Benchmark 与 Profile
共同证明 V2 整体更快且 global-load 请求显著减少，但不能从这些聚合指标精确拆分
每一种因素的耗时贡献。

## cuBLAS SGEMM 单次对照

cuBLAS 程序的五组形状均通过 CPU Reference 比较，退出码为 0；Compute Sanitizer
报告 `0 errors`、`0 bytes leaked`。在同一 GPU 上分别暖机，再交替运行三轮
`1024³` 测量：

| 轮次 | 手写 V2 latency | cuBLAS latency | cuBLAS 相对 V2 加速 |
| ---: | ---: | ---: | ---: |
| 1 | 1.504998 ms | 0.222208 ms | 约 6.77× |
| 2 | 1.504461 ms | 0.222130 ms | 约 6.77× |
| 3 | 1.504733 ms | 0.222221 ms | 约 6.77× |
| 平均 | 1.504731 ms | 0.222186 ms | 6.772× |

三轮平均性能分别为 1427.15 GFLOP/s 和 9665.25 GFLOP/s；V2/cuBLAS latency
极差占均值分别约为 0.036%/0.041%。Compute Sanitizer 插桩下的耗时不参与
性能对照。

Naive/Tiled 两版已完成 Reference、正确性、Benchmark、Profile、优化、再测量与
记录的闭环；cuBLAS 对照的测量与概念解释也已完成。`6.772×` 表示在本次
同 GPU、同 FP32 形状的测量中，手写 V2 的耗时约为 cuBLAS 的 6.772 倍；
cuBLAS 耗时约降低 85.23%，不能推广为所有 GEMM 的固定加速比。

行主序 `C=A×B` 的内存可按列主序解释为 `Cᵀ=Bᵀ×Aᵀ`。因此 cuBLAS 调用中
交换 A/B 指针并正确设置 leading dimensions；内存没有额外转置，cuBLAS 写出的
列主序 `Cᵀ` 与我们按行主序读取的 C 是同一段数据。Day 7 到此停止扩展 GEMM。
