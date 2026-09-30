# CUDA Bootcamp 学习工程

本仓库按照 `CUDA_Bootcamp_Learning_Plan.md` 推进，只学习进入 CUDALM 前必需的
CUDA 能力。每个 kernel 必须完成以下闭环：

```text
参考实现 -> 朴素 CUDA -> 正确性验证 -> 性能测试 -> 性能分析
        -> 优化 -> 再次性能测试 -> 记录结论
```

## 当前目录

```text
cuda-bootcamp/
├── CMakeLists.txt
├── CUDA_Bootcamp_Learning_Plan.md
├── CUDA_复习与面试问答.md
├── README.md
├── PROGRESS.md
├── common/
│   └── cuda_check.h
├── 01_vector_add/
│   ├── main.cu
│   └── README.md
├── 02_matrix_add/
│   ├── main.cu
│   └── README.md
├── 03_transpose/
│   ├── main.cu
│   ├── tiled.cu
│   ├── README.md
│   └── TILED_TASK.md
├── 04_reduction/
│   ├── main.cu
│   ├── v1.cu
│   ├── v2.cu
│   ├── v3.cu
│   ├── README.md
│   ├── V1_TASK.md
│   ├── V2_TASK.md
│   └── V3_TASK.md
├── 05_gemm/
│   ├── gemm_harness.h
│   ├── naive.cu
│   ├── tiled.cu
│   ├── cublas_compare.cu
│   ├── README.md
│   ├── NAIVE_TASK.md
│   ├── TILED_TASK.md
│   └── CUBLAS_TASK.md
├── 06_fp16_half2/
│   ├── vector_add_harness.h
│   ├── fp32.cu
│   ├── fp16.cu
│   ├── half2.cu
│   ├── README.md
│   └── DAY8_TASK.md
└── 07_softmax/
    ├── softmax_harness.h
    ├── v0.cu
    ├── v1.cu
    ├── v2.cu
    ├── v3.cu
    ├── v3_fp16.cu
    ├── softmax_fp16_harness.h
    ├── README.md
    ├── DAY9_V0_TASK.md
    ├── DAY9_V1_TASK.md
    ├── DAY9_V2_TASK.md
    ├── DAY9_V3_TASK.md
    └── DAY9_V3_FP16_TASK.md
```

## 当前进度

- Day 1 向量加法：已通过正确性与 Compute Sanitizer 验收。
- Day 2 矩阵加法：已通过正确性、Compute Sanitizer 和 Benchmark 验收。
- Day 3–4 矩阵转置：已完成 Naive、Shared Memory Tiled、Padding、Benchmark
  与 Nsight Compute bank-conflict 对照，验收完成。
- Day 5–6 Sum Reduction：V0–V3、Benchmark 与 Nsight Compute 对照验收完成。
- Day 7 GEMM：Naive/Tiled 两版闭环、cuBLAS 对照与结果解释已验收。
- Day 8 FP16/half2：正确性、memcheck、同配置三轮 Benchmark、Nsight Compute
  对照与 Notes 已完成，验收通过。

Day 8 的验收记录位于 `06_fp16_half2/README.md`；Day 9 Softmax 的
V0/V1/V2 已验收；V3 FP32 `float4` 已完成正确性、Benchmark、
Profile 和 Notes，且已清理过时 TODO。当前进行计划中的
FP16/half2 练习：标量 FP16 基线已通过，half2 核心仍留给学习者。
详细进度见 `PROGRESS.md`。
已学习的问题与答案持续整理在 `CUDA_复习与面试问答.md`。

## 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/softmax_fp16_scalar --correctness-only
```

通常会注释已完成练习的 CMake target；当前保留 FP16 标量/half2
两个目标用于同 dtype 对照。
需要重新构建历史练习时，取消 `CMakeLists.txt` 中对应注释。
项目级 CUDA 编译架构默认是 `sm_75`（RTX 2080 Ti）：所有通过本项目 CMake
构建的 CUDA target 都会使用它，包括重新启用的历史 target。它不影响在项目外
直接调用 `nvcc` 的命令；换 GPU 时可覆盖 CMake 变量 `CUDA_BOOTCAMP_ARCH`。

只构建当前练习：

```bash
cmake --build build --target softmax_fp16_scalar softmax_fp16_half2 -j
```

正确性完成后运行：

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  ./build/softmax_fp16_half2 --correctness-only
```

## 协作方式

请先自行完成当前阶段的 TODO，再提交源码、构建输出、正确性输出和 Sanitizer
输出供 Review。任何性能优化都必须由可复现的测量结果证明。

所有 Benchmark/Profile 的人类可读输出必须在数值后显式标注单位，例如
`0.259670 ms`、`516.88 GB/s`；后续创建的模板同样遵守此规范。
