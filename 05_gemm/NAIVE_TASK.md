# V1：Naive CUDA GEMM 任务

## 今天需要理解的最少理论

每个二维 CUDA thread 负责一个输出元素 `C[row, col]`。它沿 K 维读取：

```text
A[row, 0...K-1]
B[0...K-1, col]
```

并完成长度为 K 的点积。对于 row-major 数据：

```text
A[row, inner] -> A[row * K + inner]
B[inner, col] -> B[inner * N + col]
C[row, col]   -> C[row * N + col]
```

本阶段先建立正确基线。不要使用 shared memory、warp shuffle、向量化或任何 GEMM
库，也不要提前实现 Tiled 版本。

## 你需要完成

只修改 [naive.cu](naive.cu) 中的三个 TODO：

1. 根据二维 block/thread 得到 `row` 和 `col`。
2. 同时检查 `row < M` 与 `col < N`。
3. 使用 FP32 accumulator 遍历 K 维并写回一个 C 元素。

Host 端测试、Benchmark 和 Profile 框架已经提供，不需要重写。

## 正确性验收

测试包含方阵、非方阵以及不能整除 `16 × 16` block 的形状：

```text
(M,N,K) =
(1,1,1)
(3,5,7)
(31,33,17)
(33,31,65)
(64,48,37)
```

必须满足：

- 所有测试输出 `PASS`，程序退出码为 0。
- 输出同时报告最大绝对误差和最大相对误差。
- Compute Sanitizer 报告 `0 errors`、`0 bytes leaked`。

## Benchmark

按照学习计划测试：

```text
256³, 512³, 1024³
warmup = 5
iterations = 20
```

CUDA Event 只测 kernel，输出显式包含 `ms` 和 `GFLOP/s`。丢弃一次整程序暖机后，
连续运行三次并提交完整输出。

## 构建与运行

```bash
cmake -S . -B build
cmake --build build --target gemm_naive -j
./build/gemm_naive
echo $?

compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --error-exitcode 99 \
  ./build/gemm_naive

./build/gemm_naive >/dev/null && \
./build/gemm_naive && \
./build/gemm_naive && \
./build/gemm_naive
```

## 完成后回答

1. 对 `M=31, N=33, K=17, block=(16,16)`，grid 是多少？右下角 block 有多少
   个线程负责有效输出？
2. 当同一个 warp 的 thread 横向计算连续 `col` 时，读取 A 和 B 的访存模式分别
   是什么？哪些数据被不同线程重复读取？
3. 为什么一个输出元素需要约 `2K` FLOPs？

提交 kernel、正确性输出、退出码、Compute Sanitizer、暖机后三轮 Benchmark 和
上述回答。完成 Review 后再进入 Tiled GEMM。

## 当前验收状态

- Kernel、正确性、退出码和 Compute Sanitizer：通过。
- 三轮 Benchmark：通过，`1024³` 平均 2.348016 ms、914.59 GFLOP/s。
- `2K` FLOPs 解释：通过。严格数学计数是 K 次乘法和 K-1 次加法；性能报告按
  GEMM 惯例将每次 multiply-add 计作 2 FLOPs，因此使用 `2MNK`。
- grid 与 A/B 访存解释已完成 Review 并修正。
- V1 最小 Profile：完成。`1024³` 单 kernel 记录到 2.35 ms、67,108,864 次
  L1/TEX global-load request、134,216,439 个 sector，SM/DRAM throughput 分别
  为 62.50%/0.90%。实验条件与解释见 [README.md](README.md)。

V1 基线已验收；现在进入 V2 Tiled GEMM。以下命令保留，便于之后复查基线。

运行并提交以下原始输出：

```bash
METRICS=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum

CUDA_VISIBLE_DEVICES=0 ncu \
  --kernel-name regex:gemm_naive_kernel \
  --launch-count 1 \
  --metrics "$METRICS" \
  ./build/gemm_naive --profile
```

仅凭这组指标不能断言单一瓶颈。V2 完成后使用完全相同的 Profile 输入与指标
做单变量对照。
