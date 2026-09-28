# Day 9 当前任务：Softmax V2 Warp Shuffle

## 今天只需理解什么

V1 用 shared-memory tree 分别规约行最大值与指数和。V2 不改变数值公式、
一行一个 block 的布局或线程到列的映射，只把 block 内规约改成两层：
先在每个 warp 内使用 `__shfl_down_sync`，再让 warp 0 规约各 warp 的
partial。warp 间传值仍需少量 shared memory 和 block barrier。
“减少 shared 访问与 barrier”是待检验的优化假设，不是性能结论。

## 你需要完成的代码

文件为 `v2.cu`；共用的 CPU Reference、输入生成、17 个 shape、误差检查、
CUDA Event、Profile 模式和 launch 已复用，不要重写。当前模板保留 V1
已验收的连续列访问、局部 max/sum 扫描和最终归一化。你只负责四处 TODO：

1. 实现 `warp_reduce_max_v2`：所有参与的 lane 使用 full mask 做 warp 内
   最大值规约；不要只让 lane 0 调用 shuffle。
2. 实现 `warp_reduce_sum_v2`：同样完成 warp 内求和。
3. 让 warp 0 从 `warp_max[0..7]` 取得八个 partial max；其余 24 个 lane
   提供 `-INFINITY`，再次做 warp max。由 lane 0 写出 `row_max_shared`，
   再同步并让整个 block 读到同一个行最大值。
4. 对 `warp_sum` 做同样的跨 warp 求和；其余 lane 提供 `0`。由 lane 0
   写 `row_sum_shared`，同步后再归一化。替换模板中 `row_max=0`、
   `row_sum=1` 两个占位值。

接口保持为 `softmax_v2_kernel(const float* input, float* output,
int rows, int hidden)`，输入输出均为行主序 FP32。固定 `block=256`，
共 8 个完整 warp；`hidden<256` 时没有数据的线程仍必须参加 shuffle
和 barrier。max 的无效元素是 `-INFINITY`，sum 的无效元素是 `0`。
不要提前返回，不要把 `__syncthreads()` 放进只有部分线程进入的分支。

模板可编译，但 TODO 未完成前，partial/shared 变量可能有未使用的
编译 warning，正确性测试应出现 `FAIL`、退出码 1；这不是 V2 已通过。

## 正确性和安全验收

```bash
cmake -S . -B build
cmake --build build --target softmax_v1 softmax_v2 -j
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v2 --correctness-only
echo $?
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 ./build/softmax_v2 --correctness-only
echo $?
```

验收需要 17 个 shape 全部 `PASS`、两个退出码都是 0、Sanitizer 为
`0 errors` 且 `0 bytes leaked`。特别检查 `hidden=1/31/33/257` 的
空闲 lane、尾部列和非整 warp，以及 `rows=129` 的多 block 行映射。

## Benchmark 与 Profile

确保 `v1.cu` 使用已验收的“直接重算 `expf`”版本，而不是较慢的
output 暂存实验版。V1/V2 使用同一 GPU、同一构建配置和相同测试夹具；
先整程序暖机，再交替运行三轮：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v1 >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v2 >/dev/null
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_v1
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_v2
done
```

CUDA Event 的 `latency` 单位为 `ms`。至少记录 `(128,1024)` 和
`(128,4096)` 三轮原始值与各自平均值；速度比为
`V1 平均 latency / V2 平均 latency`。短 kernel 有波动，不能只看
一轮，也不能因为用了 shuffle 就先认定 V2 更快。

正确性通过后，使用同一组最小指标对比一次 `(128,4096)` kernel：

```bash
METRICS=gpu__time_duration.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_v1_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/softmax_v1 --profile
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_v2_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/softmax_v2 --profile
```

这组 Profile 检查 shared 指令和 barrier stall 是否按预期变化；
不能仅凭单个指标解释全部 latency 差异。ncu 时间与正常 CUDA Event
计时分开记录。如果某个指标在本机 ncu 版本不可用，先运行
`ncu --query-metrics` 核对名称，并把错误输出交给我，不要猜一个数。

## 完成后给我 Review

- `v2.cu` 的实现与 17 个 shape 的正确性、退出码、Sanitizer 原始输出。
- V1/V2 交替三轮的 Event 原始输出和两个大 shape 的平均延迟。
- 上述两次 ncu 原始输出；再用自己的话解释 max/sum 各自两处
  block barrier 的作用，以及为何 warp 0 只有八个有效 partial
  却仍可用 full mask。
