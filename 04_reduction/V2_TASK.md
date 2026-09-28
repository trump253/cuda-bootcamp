# V2：连续活跃线程的 Shared Memory Tree Reduction

## 本阶段目标

保持 V1 的输入加载、尾部补零、多轮 GPU reduction、测试尺寸和 Benchmark 方法
全部不变，只替换 block 内 tree reduction 的寻址方式，验证减少 warp divergence
是否带来性能收益。

## V1 与 V2 的唯一变量

V1 从小 stride 开始，参与线程在 warp 内交错分布：

```text
stride=1：thread 0, 2, 4, 6, ...
stride=2：thread 0, 4, 8, 12, ...
```

V2 从大 stride 开始，参与线程形成连续区间：

```text
stride=128：thread 0～127
stride=64 ：thread 0～63
stride=32 ：thread 0～31
```

从源码线程映射看，前几轮中一个 warp 通常全部参与或全部不参与。最终是否表现为
更少的硬件 divergent branch，仍需结合编译器生成方式和 Nsight Compute 验证。

## Kernel 实现要求（已完成）

`v2.cu` 已按以下要求完成：

1. stride 从 `blockDim.x / 2` 开始。
2. 每轮 stride 减半，直到完成 stride=1。
3. 只让连续的低编号线程参与当前轮加法。
4. 参与线程把自己的 shared 元素与相距 stride 的元素相加。
5. 每一轮后所有线程都必须到达 `__syncthreads()`。

不要修改 global-memory 加载、尾部补零、partial sum 写回、多轮 launch、测试尺寸、
warm-up、iterations 或有效带宽公式，否则 V1/V2 不再是单变量对照实验。

## 验收标准

- 全部正确性尺寸输出 `PASS`，程序退出码为 0。
- Compute Sanitizer 报告 `0 errors` 和 `0 bytes leaked`。
- 暖机后连续运行三轮，输出数值显式携带 `ms` 和 `GB/s`。
- 计算每个尺寸的 V1/V2 latency speedup。
- 说明 stride=128、64、32、16 时，哪些 warp 完全活跃、完全不活跃或发生分歧。
- 性能结论只依据实测；不预设 V2 一定更快。

## 运行方式

```bash
cmake --build build --target reduction_v1 reduction_v2 -j

./build/reduction_v2

compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --error-exitcode 99 \
  ./build/reduction_v2

./build/reduction_v2 >/dev/null && \
./build/reduction_v2 && \
./build/reduction_v2 && \
./build/reduction_v2
```

完成后提交 kernel、正确性与 Sanitizer 输出、三轮 Benchmark，以及线程/warp
活跃情况的解释。

## V2 验收结果

全部正确性尺寸绝对误差为 0；Compute Sanitizer 报告 `0 errors`、
`0 bytes leaked`。三轮平均及 V1 对照如下：

| N | V1 latency | V2 latency | V2 有效带宽 | V1/V2 speedup |
|---:|---:|---:|---:|---:|
| 1,024 | 0.006895 ms | 0.005464 ms | 0.750 GB/s | 1.262× |
| 16,384 | 0.007022 ms | 0.005486 ms | 11.95 GB/s | 1.280× |
| 262,144 | 0.019320 ms | 0.014094 ms | 74.40 GB/s | 1.371× |
| 1,048,576 | 0.048428 ms | 0.033143 ms | 126.55 GB/s | 1.461× |
| 16,777,216 | 0.639414 ms | 0.432722 ms | 155.09 GB/s | 1.478× |

短时的前两个尺寸仍较易受固定开销和时钟波动影响；`N=2^18` 以上三轮稳定，
且 speedup 随规模增大达到约 1.48×。因此，在其余实验条件相同的前提下，连续
活跃线程的寻址方式确实带来了可测量的性能收益；具体硬件原因见下方 Profile。

`block=256` 时，stride=128、64、32 分别有 4、2、1 个完整 warp 活跃，没有
warp 内分歧；从 stride=16 开始，warp 0 中只有部分 lane 活跃，产生一个分歧
warp。V2 不会消除最后五轮的分歧，但显著改善了前几轮。

## Nsight Compute 对照

V1/V2 均只采集 `N=2^24` 第一阶段的一次 kernel，避免混入后续小规模 pass：

```bash
METRICS=smsp__sass_average_branch_targets_threads_uniform.pct,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_uniform.sum,smsp__thread_inst_executed_pred_on_per_inst_executed.ratio,smsp__thread_inst_executed_pred_on.sum,smsp__thread_inst_executed_pred_off.sum,smsp__inst_executed.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

CUDA_VISIBLE_DEVICES=0 ncu \
  --kernel-name regex:reduce_sum_v1_kernel \
  --launch-count 1 \
  --metrics "$METRICS" \
  ./reduction_v1 --profile

CUDA_VISIBLE_DEVICES=0 ncu \
  --kernel-name regex:reduce_sum_v2_kernel \
  --launch-count 1 \
  --metrics "$METRICS" \
  ./reduction_v2 --profile
```

| 指标 | V1 交错寻址 | V2 连续活跃线程 | 结论 |
|---|---:|---:|---|
| Divergent branch targets | 0 | 0 | 编译器未生成可计数的发散分支 |
| Uniform branch targets | 4,718,592 | 4,718,592 | 相同 |
| Uniform branch 比例 | 100% | 100% | 相同 |
| Shared load bank conflict | 0 | 0 | 均无冲突 |
| Shared store bank conflict | 0 | 0 | 均无冲突 |
| Warp instructions | 129,302,528 | 53,805,056 | V2 减少 58.39% |
| Predicated-on thread instructions | 3,119,906,816 | 1,090,453,504 | V2 减少 65.05% |
| Predicated-off thread instructions | 1,007,616,000 | 621,150,208 | V2 减少 38.35% |

这说明源码中的短条件被编译器主要以 predication 等方式处理，不能把 V2 的加速
简单归因于“divergent branch 数下降”。当前 Profile 直接支持的结论是：V2 在没有
引入 shared-memory bank conflict 的情况下，显著减少了实际执行的 warp/thread
指令。结合源码可合理推断，V2 避免了 V1 的取模判断及大量交错控制工作；若要把
差异进一步归因到具体机器指令，需要进入 SASS/PTX 分析，不属于当前阶段范围。

学习者已在相同配置下亲自复现以上指标，并正确判断：当前数据不支持“V2 因硬件
divergent branch 数减少而加速”；它支持“V2 动态执行指令显著减少”的结论。
