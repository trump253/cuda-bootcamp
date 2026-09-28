# Day 7：cuBLAS SGEMM 单次对照

## 任务范围

cuBLAS 是已经实现好的 GPU 数学库。本阶段只调用它作为参考性能，不要求实现库
内部算法，也不继续优化手写 GEMM。

[cublas_compare.cu](cublas_compare.cu) 已复用 Day 7 的确定性输入、CPU Reference、
混合误差判据、CUDA Event Benchmark 和 GFLOP/s 公式。学习者只负责运行、核对
正确性与性能，并解释对照结果。

## 行主序布局

我们的矩阵是 row-major `A=(M,K)`、`B=(K,N)`、`C=(M,N)`。cuBLAS SGEMM
按 column-major 解释矩阵，因此调用时把内存中的 B 放在第一个操作数位置，A
放在第二个位置，计算：

```text
Cᵀ(N,M) = Bᵀ(N,K) × Aᵀ(K,M)
```

代码中设置 `m=N`、`n=M`、`k=K`，leading dimensions 分别为 `N`、`K`、`N`。
这是一种布局解释，不额外执行转置 kernel。五组正确性尺寸包含非方阵，专门用于
检查操作数顺序和 leading dimension。

## 公平比较条件

- 只比较单卡、FP32、`1024³`。
- cuBLAS handle 在计时前创建，A/B 的初始化、显存分配和 H2D 不在计时区。
- cuBLAS 与手写 V2 均使用相同的 CUDA Event 计时，warm-up=5、iterations=20。
- cuBLAS 由库选择 launch 配置，因此不会打印手写 kernel 的 block/grid。
- 先检查 CPU Reference 的 PASS，再比较正常运行的 `ms` 与 `GFLOP/s`；
  Compute Sanitizer 下的时间不参与性能比较。
- cuBLAS 内部实现可与手写 kernel 不同；这里只报告差距，不要求追平 cuBLAS。

## 构建与测试

```bash
cmake -S . -B build
cmake --build build --target gemm_tiled gemm_cublas -j

CUDA_VISIBLE_DEVICES=0 ./build/gemm_cublas
echo $?

CUDA_VISIBLE_DEVICES=0 compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --error-exitcode 99 \
  ./build/gemm_cublas
```

性能测量先对两个程序各做一次整程序暖机，再交替运行三轮：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/gemm_tiled >/dev/null
CUDA_VISIBLE_DEVICES=0 ./build/gemm_cublas >/dev/null

for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 ./build/gemm_tiled
  CUDA_VISIBLE_DEVICES=0 ./build/gemm_cublas
done
```

只比较两个程序输出中 `M=N=K=1024` 的行。请提交正确性、退出码、Sanitizer 和
三轮正常运行结果，并回答：cuBLAS 相对 V2 快多少倍？为什么这里要交换 A/B
的调用顺序？完成记录后 Day 7 GEMM 停止扩展。

## 当前验收状态

- 五组正确性测试、退出码与 Compute Sanitizer 已通过。
- 同 GPU 交替三轮 `1024³` 测量已完成：手写 V2 平均 `1.504731 ms`、
  `1427.15 GFLOP/s`；cuBLAS 平均 `0.222186 ms`、`9665.25 GFLOP/s`，
  cuBLAS 相对 V2 加速约 `6.772×`。
- 学习者已正确解释：`6.772×` 是本次同条件 `1024³` 的速度比；交换 A/B
  源于 `(A×B)ᵀ=Bᵀ×Aᵀ` 与 cuBLAS 的列主序内存解释。Day 7 验收完成。
