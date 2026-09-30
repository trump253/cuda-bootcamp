# Day 9 V3 补充：FP16 存储与 `half2` 成对访存

## 本次只学什么

学习计划要求尝试 FP16 的合理向量化。本次保持 V2 已验收的“一行一个
256 线程 block、两级 warp 规约、稳定 Softmax”结构，只把输入/输出
设为 FP16。求行最大值、`expf`、指数和与归一化均用 FP32；最终写回
才舍入到 FP16。不要把 `__hadd2` 当作 Softmax 的求和公式，也不要
引入新的 GEMM、极致优化或完整 FP16 数学课程。

先用同一源码构建 `softmax_fp16_scalar` 基线，再实现
`softmax_fp16_half2`。两者的 dtype、输入、规约、计时完全一致，
区别只在对齐行的成对读取和写回。**half2 是否更快必须实测**；
不能把 FP16 与 FP32 两个 dtype 的时间差当成 half2 的收益。

## 你需要完成的代码

文件为 `v3_fp16.cu`。标量 FP16 路径、两级规约、CPU Reference、
正确性、CUDA Event 和 Profile 入口都已准备好。你只填
`SOFTMAX_FP16_USE_HALF2=1` 路径的六处 TODO：

1. 对齐行的 max 阶段：一线程负责 `pair=tid, tid+256, ...`，
   每次读取两个相邻 half，分别提升为 float 后求局部最大值。
2. max 阶段用标量处理可能存在的一个尾部 half。
3. sum 阶段按相同 pair 映射读取，分别计算两个
   `expf(float(x)-row_max)` 并累加到 FP32。
4. sum 阶段处理尾部。
5. 输出阶段将两个 FP32 归一化结果分别舍入为 half，成对写回。
6. 输出阶段标量写回尾部。

可以查阅 Day 8 `half2.cu` 的成对指针转换；拆出两个 half 分量时
可查当前 CUDA 头文件中的 `__low2half`、`__high2half` 等接口。
不要求你自己实现新的 warp 规约。对 `half2` 的输入、输出行首
都需满足 4 字节对齐；框架已按实际地址判定，未对齐行走标量路径。
即使行首对齐，`hidden` 为奇数时最后一个元素也不能读完整 half2。
没有有效元素的线程仍须参与所有 shuffle/barrier。

## 正确性和安全验收

主机先把输入舍入为 half，再从这个真实 FP16 输入生成 double CPU
Reference；这样不会把输入量化误差误判成 kernel 错误。输出检查
最大绝对/相对误差和行和，允许 FP16 输出的舍入误差；漏写位置保留
NaN 哨兵。判定用“绝对误差阈值 + 相对误差项”；很小的概率即使
相对误差较大，只要绝对误差仍在 FP16 舍入范围内也可通过，不能
仅凭打印的 `max_rel_error` 单项判失败。共 19 个 shape，含
`hidden=1/2/31/33/257` 和多行边界。

```bash
cmake -S . -B build
cmake --build build --target softmax_fp16_scalar softmax_fp16_half2 -j
CUDA_VISIBLE_DEVICES=0 ./build/softmax_fp16_scalar --correctness-only
CUDA_VISIBLE_DEVICES=0 ./build/softmax_fp16_half2 --correctness-only
echo $?
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 ./build/softmax_fp16_half2 --correctness-only
echo $?
```

当前 `half2` 只是可编译模板：TODO 未填时出现 `FAIL`、退出码 1
是预期，不是验收通过。完成后需 19 组全 `PASS`、两个退出码均为 0，
Sanitizer 报告 `0 errors`、`0 bytes leaked`。特别说明奇数 `hidden`
时哪些行可走 half2、哪些行必须退回标量路径。

## Benchmark 与 Profile

正确性通过后，先整程序暖机，再在同一 GPU 上交替运行至少三轮：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/softmax_fp16_scalar >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/softmax_fp16_half2 >/dev/null
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_fp16_scalar
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_fp16_half2
done
```

CUDA Event 报告 kernel-only 平均 `latency`，单位 `ms`；至少整理
`(128,1024)`、`(128,4096)` 的三轮原始值与平均值，计算
`标量平均 latency / half2 平均 latency`。若 half2 无收益也如实记录。
两版同形状 `(128,4096)` 的最小 Nsight Compute 对照：

```bash
METRICS=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_fp16_kernel \
  --launch-count 1 --metrics "$METRICS" \
  ./build/softmax_fp16_scalar --profile
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:softmax_fp16_kernel \
  --launch-count 1 --metrics "$METRICS" \
  ./build/softmax_fp16_half2 --profile
```

L1/TEX 的 request/sector 不等于实际 DRAM 字节数，ncu 时间也不与
正常 Event 绝对时间混比。若指标不可用，先查询本机名称，不猜数值。

## 完成后给我 Review

- `v3_fp16.cu` 的 half2 路径、19 组正确性输出和退出码。
- Compute Sanitizer 错误/泄漏摘要。
- 两版交替三轮 Event 原始输出，以及两次最小 ncu 原始输出。
- 用自己的话解释为什么 max/sum 保持 FP32，以及 `hidden=33` 时
  行首对齐与单元素尾部如何处理。
