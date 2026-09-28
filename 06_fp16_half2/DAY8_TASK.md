# Day 8 任务：FP16 与 half2 向量加法

## 今天要理解的最少理论

- `__half` 是 16 位存储类型。本练习的标量版用 `__hadd` 完成一次 FP16
  加法；CPU 参考实现将已量化的 half 输入转成 float，加一次后转回 half，
  用于验证相同的单次舍入结果。这里只做一次加法，没有多项归约，不要把它
  误称为“FP32 accumulation”。多项累加时，输入/输出 dtype 与 accumulator
  dtype 可以不同；这个概念留给后续确实需要累加的算子。
- `__half2` 把相邻两个半精度值装在一个 32 位值里；`__hadd2` 对两个 lane
  分别做 FP16 加法，与标量版保持相同算术精度。`cudaMalloc` 返回的缓冲区满足
  这里的对齐要求；按 pair 访问时仍需避免奇数 `N` 的越界。
- 优化的判断依据是同尺寸的 CUDA Event 延迟、算法有效带宽，以及 Profile
  观测；“指令看起来更少”不能代替实测。

## 你需要写的代码

1. 在 `fp16.cu` 的 `vector_add_fp16_kernel` 中完成索引、边界检查、
   `__hadd` 逐元素加法。
2. 在 `half2.cu` 的 `vector_add_half2_kernel` 中完成每线程一对相邻元素的
   `__half2` 访问和 `__hadd2`；奇数长度的最后一个元素走标量路径。
3. `fp32.cu` 和 `vector_add_harness.h` 已准备好，不需重写测试、计时或
   Host/Device 内存管理。若你发现框架本身有问题，请先指出现象再修改。

## 测试与验收

完成 FP16 后先执行：

```bash
cmake --build build --target vector_add_fp16 -j
CUDA_VISIBLE_DEVICES=0 ./build/vector_add_fp16 --correctness-only
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 ./build/vector_add_fp16 --correctness-only
```

half2 完成后把命令中的 `vector_add_fp16` 换为 `vector_add_half2`，再做同样检查。
正确性测试包含 `N=1/2/3`、warp/block 边界和 `N=2^20+3`；所有用例必须
`PASS`、进程退出码为 0、Sanitizer 为 0 errors 和 0 bytes leaked。

在两种 kernel 都验收后，先分别暖机一次，再交替运行三轮：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/vector_add_fp32 >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/vector_add_fp16 >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/vector_add_half2 >/dev/null

for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/vector_add_fp32
  CUDA_VISIBLE_DEVICES=0 ./build/vector_add_fp16
  CUDA_VISIBLE_DEVICES=0 ./build/vector_add_half2
done
```

输出中的 `latency` 单位是 `ms`，`effective_bandwidth` 单位是 `GB/s`。
至少记录 `N=2^20` 和 `N=2^24` 的三轮结果。比较延迟时，用同一个 `N`；
解释带宽时，注意 FP32 与 FP16 的算法字节数不同。

最后在同一 GPU 上用 Nsight Compute 分别采集 FP16 标量和 half2 的一次
`N=2^24` kernel。先选最小指标：GPU kernel duration、global load/store
请求和 sector 数；若指标名称与当前 ncu 版本不符，先运行
`ncu --query-metrics` 查可用名称。两个程序的 `--profile` 参数都只发射一次
目标 kernel，便于你自己做单变量对照。不要拿 ncu 重放耗时与正常 Event
Benchmark 的平均延迟混为一谈。

当前 RTX 2080 Ti 环境可先用下面的最小命令，你自己运行并保留原始输出：

```bash
METRICS=gpu__time_duration.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum

CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:vector_add_fp16_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/vector_add_fp16 --profile
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name regex:vector_add_half2_kernel \
  --launch-count 1 --metrics "$METRICS" ./build/vector_add_half2 --profile
```

这些 L1/TEX 请求与 sector 数不是 DRAM 字节数；比较时同时注意 half2 的
grid 只有标量 FP16 的约一半，原始计数应结合处理元素数来解释。

## 完成后交给我 Review

- 两个 kernel 的源代码或指出已修改的文件。
- 三种版本的正确性输出、`echo $?` 和两种 FP16 版本的 Sanitizer 输出。
- 三轮 Benchmark 原始输出，以及你计算的同尺寸延迟比。
- 两种 FP16 版本的 Profile 原始指标和一句自己的解释：half2 是否真的更快？
