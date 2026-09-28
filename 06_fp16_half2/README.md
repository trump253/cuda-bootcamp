# Day 8：FP16 / half2 Element-wise Vector Add

本节只比较同一个向量加法任务的三种实现：FP32 标量、FP16 标量、
FP16 `half2`。不扩展到 GEMM、Tensor Core 或复杂算子。

## 文件分工

- `fp32.cu`：已完成的 FP32 对照组，不需要重写。
- `fp16.cu`：你实现一线程一元素的 FP16 kernel。
- `half2.cu`：你实现一线程两个元素的 `half2` kernel，以及奇数长度尾部。
- `vector_add_harness.h`：共用的确定性输入、CPU Reference、正确性验证、
  CUDA Event Benchmark 和单 kernel Profile 入口。正常情况下不要改动。
- `DAY8_TASK.md`：本节的任务、验收标准和记录表。

## 计时口径

CUDA Event 只包围重复 kernel launch；`cudaMalloc` 和 H2D/D2H 不在计时区间。
每种实现都做 10 次 warm-up、100 次正式迭代，单次延迟为总 Event 时间除以
100。有效带宽按算法字节数计算：`3 × N × sizeof(元素) / latency`，
这里的三份数据分别是读取 A、读取 B、写入 C。它不是实际 DRAM 吞吐量。

FP32 每元素 4 字节，FP16/half2 每元素 2 字节；不能只看 GB/s 判断哪个版本
延迟更低。小尺寸还可能主要受固定启动开销影响。是否加速以同尺寸、同 GPU、
多次测量为准。

## 构建

```bash
cmake -S . -B build
cmake --build build --target vector_add_fp32 vector_add_fp16 vector_add_half2 -j
```

当前 CMake 3.16 工程在 `CMakeLists.txt` 中为全项目 CUDA 目标指定 `sm_75`，
匹配 RTX 2080 Ti。VS Code 的 CMake Tools 直接配置和构建即可；
若以后换 GPU，可通过 CMake 配置变量 `CUDA_BOOTCAMP_ARCH` 覆盖目标架构。
VS Code 的 C/C++ 扩展会读取 `build/compile_commands.json` 中的实际编译参数。
本次 Debug 构建对 nvcc 传入 `-g`，没有传入会关闭设备代码优化的 `-G`。
三版使用相同构建配置，因此可以比较本次 kernel latency；若以后改用
Release，应将其视作另一组实验，不与这里的数值直接混比。

FP16 标量与 half2 kernel 已由学习者填写；当前正确性与 memcheck 已通过，
下面的性能数据是在明确记录的 Debug 构建条件下取得，已用于本节性能验收。

## 学习顺序

先读 `DAY8_TASK.md`，独立完成 `fp16.cu`，检查正确性与 Sanitizer；
再完成 `half2.cu`。最后三种版本交替运行三轮 Benchmark，并做至少一次
Nsight Compute 对照。把原始输出和自己的分析发给导师 Review，再将结论写回本页。

## 实验记录（Debug 构建）

- GPU：`CUDA_VISIBLE_DEVICES=0`，RTX 2080 Ti；编译架构 `sm_75`。
- FP16 标量与 half2 的 11 组正确性用例全部 `PASS`，最大绝对误差为 0；
  两版 memcheck 均为 `0 errors`、`0 bytes leaked`。FP32 三轮普通运行中的
  正确性用例也全部 `PASS`。
- 以下均为同一 `N`、warm-up 10 次、CUDA Event 迭代 100 次、交替运行三轮；
  有效带宽使用算法字节数计算，不是实测 DRAM 流量。

| N | 版本 | 三轮 latency（ms） | 平均 latency（ms） | 平均有效带宽（GB/s） |
| --- | --- | --- | ---: | ---: |
| `2^20` | FP32 | 0.024907 / 0.024924 / 0.025055 | 0.024962 | 504.08 |
| `2^20` | FP16 标量 | 0.014540 / 0.014500 / 0.014541 | 0.014527 | 433.09 |
| `2^20` | half2 | 0.012894 / 0.012952 / 0.012871 | 0.012906 | 487.50 |
| `2^24` | FP32 | 0.362779 / 0.362595 / 0.362712 | 0.362695 | 555.08 |
| `2^24` | FP16 标量 | 0.198042 / 0.198070 / 0.198144 | 0.198085 | 508.18 |
| `2^24` | half2 | 0.182538 / 0.182476 / 0.182422 | 0.182479 | 551.64 |

`N=2^24` 时，FP16 标量 / half2 的同尺寸延迟比约为 `1.086×`，half2
平均 latency 低约 `7.88%`。FP32 有效带宽数值略高于 half2，但它每元素
读写 12 字节、FP16/half2 每元素读写 6 字节，因此 FP32 的实际 latency
约为 half2 的 `1.988×`，不能将 GB/s 排序误读为快慢排序。`N=1024` 的
三轮结果含明显波动，暂不用于优化结论。

学习者亲自采集的 `N=2^24` Nsight Compute 对照：

| 指标 | FP16 标量 | half2 |
| --- | ---: | ---: |
| kernel duration（µs，Profile） | 200.80 | 180.83 |
| L1/TEX global-load request | 1,048,576 | 524,288 |
| L1/TEX global-store request | 524,288 | 262,144 |
| L1/TEX global-load sector | 2,097,152 | 2,097,152 |
| L1/TEX global-store sector | 1,048,576 | 1,048,576 |

half2 每个 warp 的线程各处理两个 half，单条访存指令覆盖的连续字节数
翻倍，global load/store 的请求数减半，而同尺寸总 sector 数不变。这些
L1/TEX sector 不能直接当作真实 DRAM 流量；当前数据支持“half2 在此
Debug 构建下更快”，尚不能单独证明加速全部来自请求数减少。若以后需要
Release 性能数值，可另行配置并重新采集，不作为本节验收前置条件。
