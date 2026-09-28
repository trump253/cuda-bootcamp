# Day 9 当前任务：Softmax V0

## 今天要理解的最少理论

对于一行 `x[0..hidden-1]`，先求 `m=max(x)`，再计算
`y[j]=exp(x[j]-m)/sum_k exp(x[k]-m)`。减去最大值不会改变 Softmax 结果，
却可避免大正数输入直接 `exp(x)` 溢出。本次 CPU Reference 用 `double` 计算，
GPU 输入与输出用 `float`；不要求逐位相等，需要检查误差、有限值和每行和为 1。

V0 让一个线程负责一整行，便于先隔离算法正确性。它不会充分利用行内并行，
后续 V1 才让同一行的多个线程协作做最大值与求和规约。本阶段不要提前写
Shared Memory、Warp Shuffle 或 FP16 版本。

## 你需要自己完成的代码

在 `v0.cu` 的 `softmax_v0_kernel` 中完成四处 TODO：

1. 用一维 thread 索引确定负责的 `row`；无效行不能访问输入或输出。
2. 对该行求最大值。
3. 对该行累加稳定形式的指数，得到分母。
4. 写出该行每个元素的归一化结果。

接口固定为 `softmax_v0_kernel(const float* input, float* output,
int rows, int hidden)`，行主序下元素地址为 `row * hidden + col`。
Host 侧的输入准备、Reference、内存管理、测试和计时均已给出，不需要重写。

## 验收与运行

```bash
cmake -S . -B build
cmake --build build --target softmax_v0 -j
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v0 --correctness-only
echo $?
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 ./build/softmax_v0 --correctness-only
echo $?
```

测试包含计划要求的 `rows=1/16/128` 与 `hidden=128/512/1024/4096` 的
12 种组合，另有 `hidden=1/31/33/257` 及 `rows=129` 的跨 block 边界。
输入同时包含大正数、大负数
和完全相同的一行。程序要求所有用例 `PASS`、退出码 0；Sanitizer 要求
`0 errors` 和 `0 bytes leaked`。框架报告 `max_abs_error`、
`max_rel_error`、`max_row_sum_error`；相对误差分母使用
`max(|reference|, 1e-6)`，避免把近零概率的相对误差无限放大。
判定还要求输出有限、非负、逐元素满足
`abs_error <= 2e-5 + 1e-4 * |reference|`，每行和偏差不大于 `2e-4`。

正确性通过后，先整程序暖机，再至少独立运行三次普通模式，保存原始输出：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v0 >/dev/null
for i in 1 2 3; do CUDA_VISIBLE_DEVICES=0 ./build/softmax_v0; done
```

Benchmark 输出 `latency=... ms`。只比较同一形状、同一编译条件、同一 GPU
下的结果。V0 的慢不构成错误；它是后续优化的基线。

正确性和 Benchmark 后，自行用 Nsight Compute 对 `(128,4096)` 的单次 V0
kernel 采集最小指标，完成本版的 Profile 环节：

```bash
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_v0_kernel \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  ./build/softmax_v0 --profile
```

若指标名与本机 ncu 版本不一致，先用 `ncu --query-metrics` 查询。不要把 ncu
重放期间的 duration 和普通 CUDA Event 平均延迟直接混在一起比较。

## 完成后给我 Review 的内容

- `v0.cu` 的修改、正确性输出与进程退出码。
- Compute Sanitizer 原始输出。
- 三轮 Benchmark 原始输出及你对 V0 哪部分可能慢的判断。
- Nsight Compute 原始指标，以及你对读取次数和性能限制的初步解释。

通过 V0 后再创建 V1 的 Shared Memory 模板，不提前交付最终优化代码。
