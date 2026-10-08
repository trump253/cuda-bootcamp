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
├── 07_softmax/
│   ├── softmax_harness.h
│   ├── v0.cu
│   ├── v1.cu
│   ├── v2.cu
│   ├── v3.cu
│   ├── v3_fp16.cu
│   ├── softmax_fp16_harness.h
│   ├── README.md
│   ├── DAY9_V0_TASK.md
│   ├── DAY9_V1_TASK.md
│   ├── DAY9_V2_TASK.md
│   ├── DAY9_V3_TASK.md
│   └── DAY9_V3_FP16_TASK.md
└── 08_nsys/
    ├── README.md
    └── NOTES.md
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
- Day 9 Softmax V0–V3：FP32 `float4` 与 FP16 `half2` 均完成闭环；
  向量化未获得所有形状上一致的性能收益。
- Day 10 Nsight Systems：三份时间线、传输与同步定位、依赖顺序和
  短 kernel 提交开销分析已完成 Review，验收通过。

已完成 [Day 10 Nsight Systems 时间线练习](08_nsys/README.md)，
下一阶段为 Day 11 Nsight Compute。
Bootcamp 最终独立 Softmax 验收尚未完成。
详细进度见 `PROGRESS.md`。
已学习的问题与答案持续整理在 `CUDA_复习与面试问答.md`。

## 构建与运行

```bash
cmake -S . -B build
cmake --build build --target vector_add softmax_v2 reduction_v3 -j 8
./build/reduction_v3 --timeline
```

通常会注释已完成练习的 CMake target；当前仅启用 Day 10 所需的
Vector Add、Softmax V2 与 Reduction V3。
需要重新构建历史练习时，取消 `CMakeLists.txt` 中对应注释。
项目级 CUDA 编译架构默认是 `sm_75`（RTX 2080 Ti）：所有通过本项目 CMake
构建的 CUDA target 都会使用它，包括重新启用的历史 target。它不影响在项目外
直接调用 `nvcc` 的命令；换 GPU 时可覆盖 CMake 变量 `CUDA_BOOTCAMP_ARCH`。

三个采集命令、CLI 报表与记录模板见 `08_nsys/README.md` 和
`08_nsys/NOTES.md`；Day 10 主要任务是学习者亲自分析时间线。

## 协作方式

后续核心练习仍由学习者独立完成，再提交源码、构建输出、正确性输出和
Sanitizer 输出供 Review。任何性能优化都必须由可复现的测量结果证明。

所有 Benchmark/Profile 的人类可读输出必须在数值后显式标注单位，例如
`0.259670 ms`、`516.88 GB/s`；后续创建的模板同样遵守此规范。
