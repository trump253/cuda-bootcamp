# V2：Shared Memory Tiled GEMM

## 目标

使用 `16 × 16` tile，让一个 block 协作加载 A/B 子块并在 shared memory 中复用。
V2 保持与 V1 相同的数据、正确性、Benchmark 和 Profile 框架，形成单变量对照。

## 核心关系

一个 `16 × 16` 输出 tile 包含 256 个 C 元素。每个 K 阶段只需协作加载：

```text
16 × 16 个 A 元素
16 × 16 个 B 元素
```

这些数据随后被 block 中多个线程重复使用，而不是每个线程都从 global memory
独立加载自己的 A 行和 B 列。

## 你需要完成

只修改 [tiled.cu](tiled.cu) 中的四组 kernel TODO；Host 端沿用 V1 已验收的
CPU Reference、正确性、Benchmark 和 Profile 框架。

1. 根据 `blockIdx` / `threadIdx` 确定一个线程负责的输出 `C[row,col]`，并初始化
   FP32 accumulator。
2. 将 K 维分成长度为 16 的阶段。每个线程协作加载一个 A 元素和一个 B 元素到
   `tile_a` / `tile_b`；当输出行、列或 K 的尾部越界时，相应 shared 元素写 0。
3. 等所有线程写完当前阶段的 tile，再沿 tile 内 K 维进行点积；当前阶段使用完毕
   后，再允许其他线程覆写 shared tile。
4. 所有阶段结束后，仅对有效的 `C[row,col]` 写回结果。

所有 256 个线程必须执行相同次数的 block barrier。边界线程可以贡献 0，但不要
在 barrier 前提前 return。特别检查 `K=17/37/65` 的最后一个不完整 tile。

## 验收要求

- 五组正确性尺寸 `(1,1,1)`、`(3,5,7)`、`(31,33,17)`、`(33,31,65)`、
  `(64,48,37)` 全部 PASS；程序退出码 0。
- Compute Sanitizer 输出 `0 errors`、`0 bytes leaked`。
- 丢弃一次整程序暖机后，三轮 `256³/512³/1024³` Benchmark 稳定，输出显式
  携带 `ms` 和 `GFLOP/s`，并与 V1 的同尺寸平均值比较。
- 由你运行 Nsight Compute，对照 V1 的单 kernel `1024³` Profile：观察
  global-load request/sector、SM/DRAM throughput；同样的计数变化才可作为
  “减少加载请求”的证据。
- 能解释 A/B tile 各自被哪些输出线程复用，以及为什么不完整 tile 要补 0。

## 构建与运行

```bash
cmake -S . -B build
cmake --build build --target gemm_tiled -j
./build/gemm_tiled
echo $?

compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --error-exitcode 99 \
  ./build/gemm_tiled

./build/gemm_tiled >/dev/null && \
./build/gemm_tiled && \
./build/gemm_tiled && \
./build/gemm_tiled
```

正确性、Benchmark 和内存安全通过后，再运行：

```bash
METRICS=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum

CUDA_VISIBLE_DEVICES=0 ncu \
  --kernel-name regex:gemm_tiled_kernel \
  --launch-count 1 \
  --metrics "$METRICS" \
  ./build/gemm_tiled --profile
```

提交 kernel、五组正确性结果、退出码、Sanitizer、三轮 Benchmark、Profile 原始
输出，以及两个概念问题的回答。我负责 Review 与数据整理。

## 当前验收状态

- 学习者已完成 V2 核心 kernel；五组正确性测试、退出码与 Compute Sanitizer 通过。
- 暖机后三轮 Benchmark 已完成。`1024³` 平均 1.505280 ms、1426.64 GFLOP/s，
  相对 V1 Naive 加速约 1.560×。
- 已正确解释 A/B tile 的跨线程复用、尾部补 0 和两次 block barrier 的作用。
- V2 Nsight Compute 已由学习者完成：`1024³` 单 kernel 1.51 ms，L1/TEX
  global-load request 从 V1 的 67,108,864 降到 4,194,304（减少 93.75%），
  sector 从 134,216,439 降到 16,656,832（减少约 87.59%）。
- V2 的正确性、Benchmark、Profile、Re-benchmark 和 Notes 闭环验收完成。
  下一步只做学习计划规定的一次 cuBLAS 对照。

停止边界：不做 register tiling、Tensor Core、WMMA、CUTLASS、CuTe 或 async copy。
