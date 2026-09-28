# V1：Shared Memory Tiled Transpose

## 本阶段目标

通过 shared-memory tile，把 Naive Transpose 的跨步 global write 改成合并写入。
未 padding 的 `16×16` tile 已完成正确性、Benchmark 和 Compute Sanitizer 验证。
当前通过单变量实验观察 shared-memory padding 对 bank conflict 和性能的影响。

## 最少理论

每个 block 分两步处理一个 tile：

```text
Global input 连续读取
        ↓
Shared-memory tile
        ↓
整个 block 同步
        ↓
从 tile 的转置位置读取
        ↓
Global output 连续写入
```

`__syncthreads()` 是 block 级 barrier。block 中所有仍在执行的 thread 都必须到达
同一个 barrier；不能把它放进只有部分边界内线程才会进入的条件分支。

## 已完成的基线

`tiled.cu` 已完成：

1. `transpose_tiled_kernel` 的 input 坐标、tile 写入和边界检查。
2. block 坐标交换、tile 转置读取和 output 边界检查。
3. 使用已预置的正确性与 Benchmark 测试夹具完成验证。

CPU Reference、最大误差、CUDA 内存管理、CUDA Event 和输出格式均复用自 V0。
kernel 内已完成的代码不需要重写，后续实验只修改明确指定的参数。

## 验收顺序

先完成正确性：

```text
1 × 1
15 × 17
31 × 33
32 × 33
256 × 256
1000 × 513
```

要求误差不超过 `1e-6`，程序退出码为 0，Compute Sanitizer 报告无错误和泄漏。

正确性通过后再运行三轮 Benchmark：

```text
256 × 256
1000 × 513
1024 × 1024
4096 × 4096
```

与 V0 baseline 比较 latency、有效带宽和 speedup：

```text
speedup = naive_latency / tiled_latency
```

## 下一实验：Padding

未 padding V1 与 padding V1 均已完成 Benchmark。CMake 会从同一份源代码生成：

```text
transpose_tiled_padding0：padding=0
transpose_tiled：padding=1
```

执行程序时添加 `--profile`，只会启动一次 `4096 × 4096` kernel，供 Nsight
Compute 采集。它不会运行正确性测试、warm-up 或 Benchmark 循环。

在 `build/` 目录中分别采集两个版本：

```bash
ncu \
  --kernel-name regex:transpose_tiled_kernel \
  --launch-count 1 \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./transpose_tiled_padding0 --profile

ncu \
  --kernel-name regex:transpose_tiled_kernel \
  --launch-count 1 \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./transpose_tiled --profile
```

重点比较 shared load 与 shared store 的 conflict 计数：预期 padding=1 会大幅
降低 load conflict，但由于当前一个 warp 跨两个 thread row，store conflict
可能从 0 增加为一个很小的非零值。Nsight Compute 用于验证原因，真实性能仍以
正常运行时的 CUDA Event Benchmark 为准。

## Profile 结果

`4096 × 4096`、`block=(16,16)` 时共有 65536 个 block，每个 block 8 个 warp，
总计 524288 个 warp：

| 版本 | Shared load conflict | Shared store conflict | 每 warp 总 conflict |
|---|---:|---:|---:|
| padding=0 | 3670016 | 0 | 7 |
| padding=1 | 524288 | 524288 | 2 |

Padding 使 shared load conflict 降低约 85.71%，load+store 总 conflict 降低
约 71.43%。这验证了地址推导：未 padding 的 8-way load conflict 对应每 warp
7 轮额外服务；padding=1 后，load 和 store 分别只剩一次 2-way conflict。

需要解释：

1. shared memory 如何把二维下标映射到 bank。
2. 为什么 stride=16 会让多个线程落在少量 bank。
3. 为什么 stride=17 能打散 bank 映射。
4. 当前 `16×16` block 下 padding 是否完全消除 conflict，还是主要降低冲突程度。

本阶段不使用 warp shuffle，也不进入 reduction 或 GEMM。
