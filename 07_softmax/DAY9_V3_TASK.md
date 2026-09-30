# Day 9 V3 FP32 `float4` 任务说明

本文件保留原练习要求；学习者已完成六处 TODO 的实现、正确性和
初轮 Benchmark/Profile。当前实测结论见 `README.md`，概念解释与
Notes 仍需整理。下文的“你需要完成”和“模板失败”描述原任务阶段。

## 今天只需理解什么

`float4` 一次表示四个相邻 FP32 元素。V3 保持 V2 的“一行一个 block、
256 个线程、两级 warp 规约”和稳定 Softmax 公式，只尝试改变每个线程
读取、写回一行数据的方式。线程 `tid` 处理第 `tid` 个四元素组，下一组
是 `tid + blockDim.x`；相邻 lane 对应相邻组。这里的目标是检验向量化
访问是否能减少指令/请求和延迟，不能仅凭源码认定更快。

把行首 `float*` 当作 `float4*` 前，必须满足 16 字节对齐；`hidden`
不是 4 的倍数时，相邻行的行首可能不对齐。框架已检查输入、输出
行首的实际对齐：未对齐的行走 V2 标量路径。对齐行先处理
`hidden / 4` 个完整组，再由标量路径处理 `hidden % 4` 个尾部元素。
尾部不足一个 `float4`，不能强行读取；无数据的线程仍需参加规约和
block barrier。

## 你需要完成什么

只修改 `v3.cu` 中的六处 TODO，其他测试、launch 和 V2 两级规约均已
提供。输入/输出接口保持为行主序 FP32 `[rows, hidden]`，输出仍是每行
数值稳定的 Softmax。

1. 对齐行的 max 阶段：按 `group=tid, tid+256, ...` 读取完整 `float4`，
   将四个分量并入 `thread_max`。
2. max 阶段的尾部：用当前线程对应的 `col` 读取不足四个的元素。
3. sum 阶段：按相同组映射累加四个 `expf(x-row_max)`。
4. sum 阶段的尾部：把剩余元素计入 `thread_sum`。
5. 写回阶段：对完整组写回四个归一化值；先保证正确，再从 Profile
   确认编译后是否真的采用向量化访存，不要仅从类型名推断。
6. 写回阶段的尾部：每个剩余元素都写一次，不得漏写或越界。

不要改变 V2 规约公式、block 大小、输入生成或误差阈值；这样对照才
主要反映数据访问的变化。这里先做 FP32 `float4`；计划提到的 FP16
合理向量化留在 FP32 闭环后再评估，不在本轮同时引入 dtype 变量。

## 正确性与安全验收

```bash
cmake -S . -B build
cmake --build build --target softmax_v2 softmax_v3 -j
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v3 --correctness-only
echo $?
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 ./build/softmax_v3 --correctness-only
echo $?
```

原模板虽可编译，但对齐行的六处 TODO 尚未完成，正确性测试预期
`FAIL`、退出码 1；这不是 V3 验收结果。完成后要求全部 19 个 shape
`PASS`、两个退出码均为 0，Sanitizer 为 `0 errors`、`0 bytes leaked`。
重点检查 `(1,1)`、`(1,2)`、`(1,31)`：对齐行只有尾部或有三元素
尾部；`(5,6)`：两元素尾部且不同行首对齐情况不同；以及
`(128,257)`、`(129,33)` 的多行/多 block 边界。

## Benchmark 与 Profile

正确性与内存检查通过后，在同一 GPU、同一构建配置下先整程序暖机，
再交替运行至少三轮 V2/V3。共用框架用 CUDA Event 报告 kernel-only
`latency`，单位 `ms`；不要计入分配或数据搬运。

```bash
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v2 >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v3 >/dev/null
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_v2
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_v3
done
```

至少整理 `(128,1024)` 与 `(128,4096)` 的三轮原始延迟和平均值，
计算 `V2 平均 latency / V3 平均 latency`。只有该比值大于 1 且结果
足够稳定时，才能说本次 V3 更快；变慢也要如实记录。

最小 Profile 对照沿用同一形状 `(128,4096)`，看读取请求和 sector：

```bash
METRICS=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_v2_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/softmax_v2 --profile
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_v3_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/softmax_v3 --profile
```

`request` 和 `sector` 是 L1/TEX 路径计数，不能直接当作 DRAM 字节数。
即使请求数下降，sector 总数也未必等比例下降；Profile 时长与正常
CUDA Event 时长应分开记录。如果指标不可用，先用
`ncu --query-metrics` 核对名称，不要猜测数值。

## 完成后交给我 Review

- `v3.cu` 的实现，以及 19 个 shape 的正确性输出和两个退出码。
- Compute Sanitizer 的错误、泄漏摘要。
- V2/V3 同配置交替三轮的原始 Benchmark 输出与平均值。
- 两版最小 Profile 原始输出；用自己的话解释哪些行使用 `float4`，
  哪些行走标量路径，以及尾部为何不能直接读取一个 `float4`。
