# CUDA Bootcamp 学习计划（10–14 天）

> 目标：在正式开始 CUDALM 前，用最短路径建立“能独立写、验证、Benchmark、Profile CUDA Kernel”的能力。
>
> 本计划面向 AI Infra / LLM Inference 场景，不追求学完 CUDA，也不追求在 Bootcamp 阶段打败 cuBLAS。  
> 核心原则：**学到足够开工，然后在 CUDALM 中继续按需学习。**

---

## 0. Bootcamp 完成后的能力标准

完成本阶段后，你应该可以在一个空目录中，不依赖完整教程，独立完成以下闭环：

1. 创建 `.cu` 工程并使用 `nvcc` / CMake 编译。
2. 正确使用：
   - `__global__`
   - `<<<grid, block>>>`
   - `threadIdx / blockIdx / blockDim / gridDim`
   - `cudaMalloc / cudaMemcpy / cudaFree`
   - `cudaDeviceSynchronize`
   - CUDA error checking
3. 理解：
   - Grid / Block / Thread
   - Warp（32 threads）
   - Global / Shared / Register memory
   - Coalesced memory access
   - `__syncthreads()`
4. 能独立实现：
   - Vector Add
   - Reduction
   - Matrix Transpose
   - Naive / Tiled MatMul
   - Softmax
5. 会做：
   - CPU / PyTorch Reference correctness check
   - CUDA Event latency benchmark
   - Nsight Systems 基本分析
   - Nsight Compute 基本分析
6. 能回答：
   - 为什么 GPU 需要大量线程？
   - 为什么 coalesced access 重要？
   - shared memory 的主要作用是什么？
   - reduction 为什么适合学习 warp primitive？
   - kernel 是 memory-bound 还是 compute-bound，意味着什么？

达到以上标准后，不继续“补完 CUDA”，直接进入 CUDALM。

---

# 1. 学习方法

## 1.1 每个 Kernel 必须走完整闭环

每一个练习统一遵循：

```text
理解问题
  ↓
CPU / Reference 实现
  ↓
CUDA Naive 版本
  ↓
正确性测试
  ↓
CUDA Event Benchmark
  ↓
Nsight Profile
  ↓
定位瓶颈
  ↓
优化
  ↓
再次 Benchmark
  ↓
记录结论
```

禁止只做到“代码能跑”。

---

## 1.2 Agent 的角色

Agent 不是替你完成作业，而是：

- 解释知识点
- 生成最小实验框架
- Review 你的代码
- 帮你定位 bug
- 帮你解释 Nsight 输出
- 帮你设计 benchmark
- 在你完成当前阶段后给下一阶段任务
- 维护学习记录和 TODO

Agent **不应该默认直接给最终优化版本**。

推荐交互方式：

> 我先实现，你先给我任务约束和接口。  
> 我写完后你 review；除非我明确要求，不要直接给完整答案。

---

# 2. 推荐目录结构

Bootcamp 不需要做成“简历项目”，但建议保持工程化：

```text
cuda-bootcamp/
├── CMakeLists.txt
├── README.md
├── notes/
│   ├── 01_programming_model.md
│   ├── 02_memory.md
│   ├── 03_reduction_warp.md
│   ├── 04_profiling.md
│   └── 05_softmax.md
│
├── common/
│   ├── cuda_check.h
│   ├── timer.h
│   └── utils.h
│
├── 01_vector_add/
├── 02_matrix_add/
├── 03_reduction/
├── 04_transpose/
├── 05_matmul/
├── 06_softmax/
│
└── benchmarks/
```

每个练习目录建议：

```text
problem/
├── main.cu
├── kernel.cu
├── kernel.cuh
├── reference.cpp
└── README.md
```

---

# 3. Day 1：CUDA Programming Model

## 学习目标

理解：

- Host vs Device
- Kernel
- Grid / Block / Thread
- SIMT 基本思想
- Kernel launch
- GPU memory allocation
- H2D / D2H

## 必须完成

### Task 1：Hello CUDA

完成最小程序：

```cpp
__global__ void hello();
```

观察：

- 多个 block
- 多个 thread
- thread id

### Task 2：Vector Add

输入：

```text
A[N]
B[N]
```

输出：

```text
C[i] = A[i] + B[i]
```

要求：

- N 不是 block size 的整数倍
- 正确处理边界
- CPU reference
- 最大误差检查

### Task 3：CUDA Error Check

实现：

```cpp
CUDA_CHECK(...)
```

并在每个 CUDA API 与 kernel launch 后使用。

## 必须理解

```text
global_tid = blockIdx.x * blockDim.x + threadIdx.x
```

能够手动画出：

```text
Grid
 ├ Block 0
 │  ├ Thread 0
 │  ├ Thread 1
 │  ...
 └ Block 1
```

## 当日验收

你能独立解释：

> N=1000、block=256 时，为什么 grid 要是 `(1000+255)/256`？

---

# 4. Day 2：二维索引与基础性能测量

## 学习内容

- 2D Grid
- 2D Block
- CUDA Event
- Warm-up
- 为什么不能简单用 CPU wall-clock 测单个 kernel

## 练习

### Matrix Add

```text
C[M, N] = A[M, N] + B[M, N]
```

使用二维：

```cpp
dim3 block(...)
dim3 grid(...)
```

## Benchmark

实现通用计时器：

```cpp
float benchmark_kernel(...);
```

规则：

- warm-up
- 至少执行几十/几百次
- 平均 latency
- 不把内存分配计入 kernel latency

## 输出

记录：

```text
Shape:
Block:
Latency:
Effective Bandwidth:
```

---

# 5. Day 3–4：GPU Memory Hierarchy

## 必须学习

### Register

线程私有，快，但数量有限。

### Global Memory

容量大、延迟高。

### Shared Memory

block 内共享，可显式管理。

### Cache

知道 L1/L2 的存在即可，暂时不深挖策略。

---

## 核心概念：Coalescing

理解：

```text
Good:
thread 0 -> x[0]
thread 1 -> x[1]
thread 2 -> x[2]

Bad:
thread 0 -> x[0]
thread 1 -> x[stride]
thread 2 -> x[2*stride]
```

---

## 练习：Matrix Transpose

### V0：Naive

```text
out[col][row] = in[row][col]
```

### V1：Shared Memory Tiled Transpose

要求使用：

```cpp
__shared__
__syncthreads()
```

### 分析

比较：

- naive
- tiled

解释：

- 哪一侧发生非合并访问？
- shared memory 如何改变访存模式？
- 是否出现 bank conflict？

---

# 6. Day 5–6：Reduction 与 Warp

这是进入 LLM Kernel 前最关键的基础。

## 学习内容

- Tree Reduction
- Shared Memory Reduction
- Warp
- Warp-level execution
- `__shfl_down_sync`
- Warp Reduction

---

## 练习：Sum Reduction

输入：

```text
x[N]
```

输出：

```text
sum(x)
```

### V0

非常朴素实现。

### V1

Shared memory tree reduction。

### V2

减少 divergence。

### V3

Warp shuffle。

---

## 要记录

每个版本：

```text
N
Block Size
Latency
Effective Bandwidth
```

并写结论：

- 哪一版本最快？
- 为什么？
- 数据规模小时为什么表现不同？
- reduction 更像 memory-bound 还是 compute-bound？

---

# 7. Day 7：Naive GEMM 与 Tiled GEMM

注意：

**本阶段不死磕 GEMM。**

## V0：CPU Reference

```text
C[M,N] = A[M,K] × B[K,N]
```

## V1：Naive CUDA GEMM

一个 thread 计算一个输出元素。

## V2：Shared Memory Tiled GEMM

例如 tile：

```text
16×16
或
32×32
```

必须理解：

- 为什么 A/B tile 能复用
- 为什么 shared memory 减少 global memory traffic

## Benchmark

测试：

```text
256³
512³
1024³
```

同时和 cuBLAS 做一次对照即可。

## 停止条件

做到：

```text
Naive GEMM
+
Tiled GEMM
```

就停止。

以下内容留给 CUDALM：

- Register tiling
- Tensor Core
- WMMA
- CUTLASS
- CuTe
- async copy

---

# 8. Day 8：FP16 与 Vectorization

## 学习

- `half`
- `__half`
- `half2`
- float accumulation
- vectorized load/store

理解：

> 存储 dtype 和 accumulator dtype 可以不同。

## 练习

将前面的 Vector Add 或简单 element-wise kernel 改成：

- FP32
- FP16
- half2

比较：

```text
latency
memory throughput
```

---

# 9. Day 9：Softmax V0–V3

这是 Bootcamp 的核心毕业练习。

目标：

```text
input  [rows, hidden]
output [rows, hidden]
```

对每一行：

```text
softmax(x)
```

---

## V0：Naive

先保证正确。

---

## V1：Shared Memory

流程：

```text
row
 ↓
reduce max
 ↓
exp(x-max)
 ↓
reduce sum
 ↓
normalize
```

---

## V2：Warp Reduction

使用：

```cpp
__shfl_down_sync
```

实现：

- warp max
- warp sum

---

## V3：Vectorized Load

尝试：

- float4（FP32）
- half2 / 合理向量化（FP16）

---

## Correctness

对照：

- CPU reference
或
- PyTorch reference

测试：

```text
rows = 1 / 16 / 128
hidden = 128 / 512 / 1024 / 4096
```

检查：

```text
max_abs_error
max_relative_error
```

---

# 10. Day 10：Nsight Systems

目标：

> 学会从系统时间线看“时间花在哪里”。

## 要做

运行：

- Vector Add
- Softmax
- 多 kernel sequence

观察：

```text
CPU launch
cudaMemcpy
kernel
gap
kernel
```

## 至少回答

1. kernel launch 是否有 gap？
2. H2D/D2H 在哪里？
3. 单次 kernel 是否太短导致 launch overhead 显著？
4. 多 kernel pipeline 的执行顺序如何？

---

# 11. Day 11：Nsight Compute

目标：

> 学会分析单个 kernel。

第一阶段只关注：

```text
Kernel Duration
Memory Throughput
DRAM Throughput
Compute Throughput
Occupancy
Warp Stall 原因（了解）
```

选择：

```text
softmax_v1
softmax_v2
softmax_v3
```

进行对比。

## 输出一份分析笔记

```markdown
# Softmax Nsight Analysis

## V1
...

## V2
...

## V3
...

## Bottleneck
...

## Why V3 wins / loses
...
```

---

# 12. Day 12–14：综合与补缺

如果前面进度快，进行以下工作。

## A. RMSNorm

提前体验 CUDALM 的第一个真实 operator。

实现：

```text
mean(x²)
rsqrt
x * weight
```

要求：

- shared reduction
- warp reduction
- FP16 input
- FP32 accumulate

---

## B. 简单 Kernel Fusion

例如：

```text
SiLU + multiply
```

实现：

```text
SwiGLU
```

理解 kernel fusion 主要减少：

- launch
- intermediate global memory traffic

---

## C. 复盘

将以下问题写入 `notes/bootcamp_summary.md`：

1. CUDA execution model 是什么？
2. global/shared/register 分别解决什么问题？
3. coalescing 为什么重要？
4. warp reduction 是怎么工作的？
5. 什么是 memory-bound？
6. 什么是 compute-bound？
7. 如何 benchmark CUDA kernel？
8. Nsight Systems 和 Nsight Compute 分别解决什么问题？
9. 为什么不能只看 occupancy？
10. 什么时候应该开始 CUDALM？

---

# 13. Bootcamp 最终验收任务

Agent 必须进行一次“毕业验收”。

## 任务

从空文件开始独立实现：

```cpp
softmax_cuda(input, output, rows, hidden);
```

要求：

- 不复制之前完整实现
- correctness
- CUDA error check
- event benchmark
- 至少两版优化
- Nsight Compute 分析
- README 解释优化过程

---

## 通过标准

同时满足：

- [ ] 能独立写 kernel
- [ ] 能处理边界
- [ ] 会 shared memory
- [ ] 会 reduction
- [ ] 理解 warp
- [ ] 至少使用一次 shuffle
- [ ] 会 benchmark
- [ ] 会 correctness check
- [ ] 会使用 Nsight Systems
- [ ] 会使用 Nsight Compute
- [ ] 能解释优化原因

完成后立即进入 CUDALM。

---

# 14. Bootcamp 阶段明确禁止

禁止：

- 花一周研究 PTX
- 试图击败 cuBLAS
- 深挖 CUTLASS 模板
- 深挖 CuTe
- 深挖 Tensor Core 指令
- 学完整 FlashAttention
- 做大规模 CUDA 题库
- 为了“代码优雅”做复杂 framework
- Agent 一次性把所有 kernel 写完

本阶段目标不是成为 CUDA 专家，而是：

> 获得完成 CUDALM 所需的最小 CUDA 独立开发能力。

---

# 15. Agent 每次工作流程

每个学习 session 建议遵循：

```text
1. 查看当前 README / TODO / progress
2. 确认今天唯一主要目标
3. 讲解最必要理论
4. 给接口和验收要求
5. 让我先实现
6. Review
7. Debug
8. Benchmark
9. Profile
10. 写学习笔记
11. 更新 progress
12. 给下一任务
```

Agent 不要一次给未来三天全部代码。

---

# 16. 推荐进度文件

创建：

```text
PROGRESS.md
```

格式：

```markdown
# Progress

## Current Stage
Day X

## Completed
- ...

## Current Task
- ...

## Problems
- ...

## Key Takeaways
- ...

## Next
- ...
```

Agent 每个 session 结束前维护一次。

---

# 17. 初始 Prompt：CUDA Bootcamp Agent

将下面内容直接作为本地 Agent 的初始 Prompt。

```text
你现在是我的 CUDA / GPU Programming 学习导师和代码 Review Agent。

我的长期目标是进入 AI Infra / LLM Inference 方向，并在完成本阶段后实现一个名为 CUDALM 的 C++/CUDA LLM inference & serving engine。

当前阶段不是让我“学完 CUDA”，而是执行仓库中的《CUDA Bootcamp 学习计划》，在 10–14 天内让我达到能够独立开发、验证、Benchmark 和 Profile 基础 CUDA kernel 的水平。

【你的职责】

1. 首先阅读当前仓库中的：
   - CUDA_Bootcamp_Learning_Plan.md（如果文件名略有不同，找到对应计划）
   - README.md
   - PROGRESS.md（若存在）
   - 当前源代码

2. 严格按照学习计划推进，不要擅自把学习范围扩展成完整 CUDA 课程。

3. 你是导师，不是代写工具。
   默认工作方式：
   - 先解释当前任务所需的最少理论；
   - 给出明确接口、输入输出、测试要求和验收标准；
   - 让我先实现；
   - 再 Review 我的代码并指出 bug、性能问题和 CUDA 设计问题；
   - 除非我明确要求“直接给完整实现”，否则不要直接提供最终优化版完整代码。

4. 每个 CUDA kernel 必须完成以下闭环：
   Reference -> Naive CUDA -> Correctness -> Benchmark -> Profile -> Optimization -> Re-benchmark -> Notes。

5. 所有性能优化必须有证据。
   不允许仅仅声称“更快”。
   尽量使用：
   - CUDA Event
   - Nsight Systems
   - Nsight Compute
   - 可复现 benchmark

6. 如果我的实现存在问题，请优先让我理解：
   - 问题发生在哪里
   - 为什么错误/慢
   - 应该验证什么
   然后再给修复建议。

7. 学习重点依次包括：
   - CUDA programming model
   - thread/block/grid
   - memory hierarchy
   - coalescing
   - shared memory
   - reduction
   - warp / shuffle
   - basic GEMM
   - FP16 / half2
   - softmax
   - benchmarking
   - Nsight profiling

8. 以下内容不是本阶段主线，不要主动带我深入：
   - PTX
   - CUTLASS / CuTe 深层实现
   - 极致 GEMM 优化
   - FlashAttention 完整实现
   - Hopper TMA
   - 分布式 CUDA

9. 维护 PROGRESS.md。
   每个 session 结束前更新：
   - 已完成内容
   - 当前问题
   - 今日关键知识
   - 下一任务

10. 当且仅当我通过 Bootcamp 最终 Softmax 验收后，明确告诉我：
   “CUDA Bootcamp 已达到进入 CUDALM 的最低能力门槛。”
   然后停止继续扩展 Bootcamp 内容。

【第一步】

现在先检查当前仓库内容。
如果仓库还是空的：
1. 创建最小 CUDA Bootcamp 工程结构；
2. 创建 CMakeLists.txt；
3. 创建 README.md；
4. 创建 PROGRESS.md；
5. 准备 Day 1 的 Vector Add 任务框架。

不要直接实现 Vector Add 的最终答案。
先告诉我：
- 今天要理解什么；
- 我要自己完成哪些代码；
- 如何判断完成；
- 完成后我应该把什么结果给你 Review。
```

---

# 18. Bootcamp 结束后的衔接

最终你应该带着以下能力进入 CUDALM：

```text
CUDA Programming Model
        ↓
Memory Hierarchy
        ↓
Reduction
        ↓
Warp
        ↓
Softmax
        ↓
Benchmark
        ↓
Nsight
        ↓
CUDALM
```

不要求：

```text
Tensor Core 专家
CUTLASS 专家
FlashAttention 专家
```

这些将在 CUDALM 中按需求学习。
