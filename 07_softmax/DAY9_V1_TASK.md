# Day 9 当前任务：Softmax V1 Shared Memory

## 今天需要理解的最少理论

V0 的一个线程独占一行，warp 内相邻线程却读取相隔一整行的地址。
V1 改为**一个 block 负责一行**，同一 warp 的相邻线程优先处理相邻列：
每个线程可从 `col=threadIdx.x` 开始，以 `blockDim.x` 为步长处理多个元素。
这样需要两次 block 内协作规约：先得到整行 `max`，再得到
`sum(expf(x-max))`。两次规约之间及每个 tree 阶段的 shared-memory 写后读
要正确同步。这里先只用 shared memory，不提前使用 warp shuffle。

## 你要自己完成的代码

在 `v1.cu` 的 `softmax_v1_kernel` 中完成五组 TODO：

1. 线程求自己负责列的局部最大值，再做 shared-memory tree max 规约。
2. 每线程求自己负责列的局部指数和，再做 shared-memory tree sum 规约。
3. 用整行指数和归一化并写出自己负责的列。

接口仍为 `(const float* input, float* output, int rows, int hidden)`；
框架已给出 `row`、`tid`、两个 shared 数组，以及每行一个 block、每 block
256 个线程的 launch。无元素线程对 max 应贡献 `-∞`，对 sum 应贡献 `0`；
**不能因 `tid >= hidden` 而提前返回**，否则部分线程可能无法到达
`__syncthreads()`。输入输出仍为行主序 FP32；无需重写 CPU Reference、
CUDA 分配或 CUDA Event 代码。

## 正确性与安全验收

```bash
cmake -S . -B build
cmake --build build --target softmax_v0 softmax_v1 -j
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v1 --correctness-only
echo $?
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 ./build/softmax_v1 --correctness-only
echo $?
```

V0/V1 共用 17 个 shape，用相同 CPU Reference 检查最大绝对误差、最大
相对误差及每行和。需要所有用例 `PASS`、正常退出码 0，Sanitizer 报告
`0 errors`、`0 bytes leaked`。特别留意 `hidden=1/31/33/257`、
`rows=129` 和 `hidden=4096`：它们分别检查空闲线程、非整块列与多轮列扫描。

## Benchmark 与 Profile

通过正确性后，在同一 GPU、同一构建配置下先暖机，再交替运行三轮：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v0 >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v1 >/dev/null
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_v0
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_v1
done
```

输出的 `latency` 单位为 `ms`。至少记录 `(128,1024)` 和 `(128,4096)`
的三轮原始数据，分别求平均；同形状加速比定义为
`V0 平均 latency / V1 平均 latency`。小规模下不保证 V1 更快，不能把
单次短 kernel 波动当成优化证据。

Profile 由你自己运行。先采集与 V0 相同的单次 `(128,4096)` kernel 指标：

```bash
METRICS=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_v1_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/softmax_v1 --profile
```

先计算 V1 的 `sector/request`，再与 V0 的 `32 sector/request` 对照；
若要进一步讨论 shared memory 或 barrier 的代价，再选相关指标，避免一次
收集过多指标。ncu 的重放耗时与正常 CUDA Event 的平均延迟分开记录。

## 完成后交给我 Review

- `v1.cu` 的代码；全部正确性用例、退出码和 Sanitizer 原始输出。
- V0/V1 交替三轮的原始 Benchmark 输出及同形状平均 latency、加速比。
- V1 的 Nsight Compute 原始指标，外加你对 warp 内相邻 lane 地址间距、
  `sector/request` 和 `__syncthreads()` 作用的解释。

本任务只到 V1；通过 Review 后再准备 V2 Warp Shuffle 模板。
