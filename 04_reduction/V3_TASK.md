# V3：Warp Shuffle Reduction

## 本阶段目标

使用 `__shfl_down_sync` 在 warp 内直接交换寄存器值，减少 V2 对 shared memory
和 block barrier 的依赖。V3 仍然沿用已经验证的多轮 GPU reduction，当前只优化
每个 block 内部的归约方式。

本版本不使用 `atomicAdd`，不做循环展开，也不进入更复杂的向量化加载。

## 最少理论

### `__shfl_down_sync` 做什么？

```cpp
received = __shfl_down_sync(mask, value, delta);
```

同一个 warp 中，每个参与 lane 从编号更高 `delta` 的 lane 读取其寄存器 `value`。
它只负责交换值，不会自动做加法，也不能跨 warp 通信。

对于 32-lane warp reduction，delta 依次为：

```text
16, 8, 4, 2, 1
```

最终只有 lane 0 的结果代表整个 warp 的和，其他 lane 的最终值不作为输出使用。

### Mask 表示什么？

mask 的每一 bit 对应一个 lane。mask 中声明参与的所有 lane 都必须执行同一个
`__shfl_down_sync`，否则行为未定义。

当前 kernel 固定 `block=256`，每个 warp 始终拥有 32 个真实线程；越界线程读取
加法单位元 0，并继续参加 shuffle。因此当前可以使用：

```cpp
kFullWarpMask = 0xFFFFFFFFU
```

不要让尾部线程提前 return。

## Block 内两级归约

```text
256 个线程
  ↓ 每个 warp 使用 shuffle
8 个 warp partial sums
  ↓ lane 0 写入 shared memory
warp_sums[8]
  ↓ 一次 __syncthreads()
warp 0 再使用 shuffle
1 个 block partial sum
```

Shuffle 不能跨 warp，因此仍需要少量 shared memory 在 8 个 warp 之间传递结果。
这里只需要一次 block barrier：保证所有 warp leader 已写完 `warp_sums`，warp 0
才能读取。

## 已完成的实现内容

学习者已在 [v3.cu](v3.cu) 中完成以下四部分：

1. 在 `warp_reduce_sum` 中实现 delta 逐轮减半的寄存器归约。
2. 让每个 warp 的全部 lane 使用 full mask 调用该函数。
3. 让每个 warp 的 lane 0 写入自己的 `warp_sums[warp_id]`。
4. 同步后，只让 warp 0：
   - 前 8 个 lane 读取 8 个 warp partial sums；
   - 其余 lane 使用 0；
   - 全部 32 个 lane 再做一次 shuffle reduction；
   - 最终由 lane 0 写入 `block_sums[blockIdx.x]`。

不要修改测试规模、warm-up、iterations、多轮 launch 或有效带宽公式。

## 验收标准

- 全部正确性测试输出 `PASS`，程序退出码为 0。
- Compute Sanitizer 报告 `0 errors` 和 `0 bytes leaked`。
- 暖机后连续运行三轮，输出显式包含 `ms`、`GB/s`。
- 与 V2 比较每个尺寸的 latency 和 speedup。
- 能解释 full mask 为什么在当前尾部 block 中仍然安全。
- 能解释为什么 warp 内不需要 `__syncthreads()`，但 warp 间仍需要一次。
- 能解释第二级归约为什么让 warp 0 的 32 个 lane 全部调用 shuffle，而不是只让
  前 8 个 lane 调用。

## 运行方式

```bash
cmake --build build --target reduction_v2 reduction_v3 -j
./build/reduction_v3
echo $?

compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --error-exitcode 99 \
  ./build/reduction_v3

./build/reduction_v3 >/dev/null && \
./build/reduction_v3 && \
./build/reduction_v3 && \
./build/reduction_v3
```

完成后提交 kernel、正确性与 Sanitizer 输出、三轮 Benchmark，以及上述三个概念
问题的回答。先由你解释数据，我再 Review 和整理。

## 当前实验结果

- 全部规定尺寸均通过 CPU Reference 对比，最大绝对误差为 0，退出码为 0。
- Compute Sanitizer：`0 errors`、`0 bytes leaked`。
- 暖机后三轮 Benchmark 平均值：

| N | V3 latency | V3 有效带宽 | V2 latency | V3 相对 V2 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.004902 ms | 0.84 GB/s | 0.005464 ms | 1.115× |
| 16,384 | 0.004961 ms | 13.21 GB/s | 0.005486 ms | 1.106× |
| 262,144 | 0.009718 ms | 107.90 GB/s | 0.014094 ms | 1.450× |
| 1,048,576 | 0.019637 ms | 213.59 GB/s | 0.033143 ms | 1.688× |
| 16,777,216 | 0.247308 ms | 271.36 GB/s | 0.432722 ms | 1.750× |

Benchmark 已证明 V3 整体实现更快；大尺寸收益更稳定。其原因不能仅写成“shuffle
一定比 shared memory 快”，还需要结合 V2/V3 Profile 对照检查 shared-memory
指令、block barrier 等变化。

## V2/V3 Profile 结果

学习者在相同 `N=2^24`、`grid=65536`、`block=256` 条件下完成了单 kernel 对照：

| 指标 | V2 | V3 | 变化 |
| --- | ---: | ---: | ---: |
| Warp instructions | 53,805,056 | 17,956,864 | 减少 66.63% |
| Shared load instructions | 8,454,144 | 65,536 | 减少 99.22% |
| Shared store instructions | 4,718,592 | 524,288 | 减少 88.89% |
| Barrier stall / active warp | 22.41% | 6.39% | 降低 16.02 个百分点 |
| Warps stalled at barrier | 246,446,359 | 31,432,461 | 减少 87.25% |

这组 Profile 直接支持：V3 显著减少了 shared-memory 指令、总执行指令和 CTA
barrier 等待，方向与 Benchmark 的 1.750× 大尺寸加速一致。它不能进一步拆分出
每一项各自贡献了多少 latency。

源码中的 block barrier 从 9 次降到 1 次，不要求动态 barrier stall 指标恰好按
9:1 下降。一次 barrier 的等待量取决于 warp 到达时间、调度以及可隐藏延迟；百分比
指标还有 active-warp 周期这一分母。Shuffle 自身有指令代价，但不是 barrier stall
未严格按 9:1 变化的直接解释。
