# 最终独立 Softmax 验收

状态：2026-10-09 最终独立 Softmax 验收通过。三版 kernel、Event、23 个 shape 正确性、memcheck、同条件三轮复测、Profile、实现解释及独立实现确认均已完成 Review。对应 [学习计划第 13 节](../CUDA_Bootcamp_Learning_Plan.md)，Bootcamp 到此结束。

本轮数据、波动范围与 Review 见 [NOTES.md](NOTES.md)：第 3.4 节为复测，第 5 节分别保存学习者原回答、导师反馈、最终修订与验收结论。下方原任务流程仅供复查；全部必修交付已完成，不再追加 kernel、测量或复盘题。Nsight Systems 能力沿用已通过的 Day 10。

## 1. 最少理论与接口

对行主序 FP32 矩阵的每一行计算：`y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`。减去行最大值保持数学结果不变，并避免大正数直接求指数溢出。涉及跨线程读写时，你需要自己设计规约、单位元与同步。

三版各自提供同一个主机接口：

```cpp
void softmax_cuda(const float* input, float* output, int rows, int hidden);
```

- input/output 是调用者已分配的 GPU 指针，均为连续行主序 `[rows, hidden]` FP32；二者不重叠。接口只负责启动计算，不在其中分配、复制或做设备全局同步。
- 本次测试只传入正 rows/hidden、有限 FP32 输入及有效指针；不扩展到原地运算、非连续 stride、空矩阵或 NaN/Inf 输入语义。
- 输出与输入同形状；必须覆盖每个元素、结果有限且非负，每行和接近 1。
- 三个实现分别编译为三个程序，同名 `softmax_cuda` 不会相互链接。启动配置已按映射补齐：naive 为 block=128、grid=ceil(rows/128)，shared/warp 为 block=256、grid=rows。

## 2. 你需要完成什么

| 文件 | 你的任务 |
| --- | --- |
| [naive.cu](naive.cu) | 从空 kernel 独立实现稳定朴素版及启动配置，先通过正确性。 |
| [softmax_final_harness.h](softmax_final_harness.h) | 独立补齐四个 Event 步骤，完成朴素版 Benchmark 基线。 |
| [shared.cu](shared.cu) | 实现优化版一，展示 shared memory 与 block 内规约；说明边界及同步。 |
| [warp.cu](warp.cu) | 实现优化版二，至少使用一次 shuffle；说明参与 mask 与跨 warp 交接。 |
| [NOTES.md](NOTES.md) | 保存测试证据、同配置三轮结果、Profile 和优化解释；不要先填猜测结论。 |

CPU double Reference、确定性输入、CUDA 错误检查、NaN 输出哨兵、结果比较、分配/复制、暖机和重复循环已提供，只复用主机支撑，不引用旧 GPU kernel。除 Event 的四个关键步骤外，重复主机代码不用再从头写。

不要复制 Day 9 的完整 kernel。可以查看 CUDA API 名称、自己的知识笔记和测试支撑；不确定时先解释思路或提交最小代码给导师 Review。三个版本按顺序完成，不要一次改多个优化变量。

## 3. 验证与测量口径

- 正确性共 23 个 shape：计划中的 `rows=1/16/128 × hidden=128/512/1024/4096`，加上 `(1,1)、(1,2)、(1,31)、(5,6)、(16,33)、(128,257)、(129,33)、(1,255)、(16,256)、(3,513)、(16,4097)`。覆盖小行、非整 block、奇数尾部与超过 4096 的尾部。
- 输入包含约 ±100 的行偏移、负数和全相等行，要求实现数值稳定；CPU 使用同一 FP32 输入并以 double 计算参考值。
- 每个元素要求 `abs_error <= 2e-5 + 1e-4 * abs(reference)`，每行和误差 `<= 2e-4`，拒绝 NaN/Inf、漏写、负数和超过允许范围的概率。最大相对误差使用 `max(abs(reference), 1e-6)` 作分母，是诊断值，不单独作为整行的硬阈值。
- Kernel-only Event：计时外完成分配、传输与暖机；start/stop 包住同一 stream 上的 100 次 launch，等待 stop 后求平均；warmup=10。统一输出 `latency=... ms`，误差和概率为无量纲。
- 本轮使用现有 Debug/sm_75 配置，不强制切换 Release；Debug 并不自动等于设备 `-G`。如你切换配置，要三版一起重新构建、验证与测量。
- 比较同一 GPU、输入 shape、构建、数学函数和计时口径，记录实际 block/grid、暖机、迭代次数；不要直接拼接 Day 9 历史时间。
- 正确性失败时默认模式不进入 Benchmark；Event TODO 未完成时拒绝输出性能数字。`--profile` 单次启动 `(128,4096)` 并检查输出，不进行 Event 测量。

## 4. 先完成朴素版

在仓库根目录运行。VS Code CMake Tools 默认也只构建这三个新目标；旧目标可取消注释复查。

```bash
cmake -S . -B build
cmake --build build --target softmax_final_naive -j 8
CUDA_VISIBLE_DEVICES=0 ./build/softmax_final_naive --correctness-only
echo $?
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 ./build/softmax_final_naive --correctness-only
echo $?
```

尚未实现的空框架应 FAIL、退出码 1，不是工程损坏。填好朴素版后，要求全部 PASS、退出码 0，memcheck 为 0 errors/0 bytes leaked。先提交朴素版代码与这两份输出给导师 Review，然后补齐 Event TODO，采集基线：

```bash
(
  set -e
  CUDA_VISIBLE_DEVICES=0 ./build/softmax_final_naive >/dev/null
  for round in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 ./build/softmax_final_naive
  done
)
```

暖机不会消除每次 launch 的稳态成本。Event 核心步骤对应 [CUDA 11.8 GPU Timers](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers)，不要把 ncu 或 Sanitizer 下的时间混入普通 Event 基线。

朴素版 Benchmark 基线完成后，先按第 6 节命令只采集 naive（暂时把版本循环改成 `for v in naive`），填写基线 Profile 与 Notes，再开始 shared 优化。每完成一版优化，都先验证、测量、Profile 与记录，然后再进入下一版；最后做三版完整对照，保持计划要求的闭环顺序。

## 5. 完成两版优化后做同条件对照

先对 shared/warp 分别构建、验证、memcheck；命令中的版本名替换即可。三版均通过后再执行下面这一组，失败时停止，不继续收集性能：

```bash
cmake --build build --target softmax_final_naive softmax_final_shared softmax_final_warp -j 8 &&
(
  set -e
  for v in naive shared warp; do
    CUDA_VISIBLE_DEVICES=0 "./build/softmax_final_${v}" --correctness-only
    CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --leak-check full \
      --error-exitcode 99 "./build/softmax_final_${v}" --correctness-only
    CUDA_VISIBLE_DEVICES=0 "./build/softmax_final_${v}" >/dev/null
  done
  for round in 1 2 3; do
    for v in naive shared warp; do
      CUDA_VISIBLE_DEVICES=0 "./build/softmax_final_${v}"
    done
  done
)
```

记录三轮原始输出、每个 shape 的平均值与极差；`加速比=前版平均延迟/后版平均延迟`，`延迟降低比例=1-后版/前版`。没有指定加速倍数，不要求所有 shape 都获益；若差异与波动接近，写“收益不稳定”，并结合证据解释。

## 6. 你亲自采集 Nsight Compute

等三版正确性及 Benchmark 完成后再采集。只分析已学的六类指标与 shared/request/sector 计数，不深入新指令或高级优化。

```bash
(
  set -e
  FINAL_METRICS=l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum
  for v in naive shared warp; do
    CUDA_VISIBLE_DEVICES=0 ncu \
      --kernel-name "regex:softmax_final_${v}_kernel" \
      --launch-count 1 \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section Occupancy \
      --section WarpStateStats \
      --metrics "$FINAL_METRICS" \
      --page raw \
      --export "build/final_softmax_${v}" \
      "./build/softmax_final_${v}" --profile
  done
)
```

再次采集时更换 export 前缀，保留上一组报告，不强制覆盖。报告保存在被 Git 忽略的 build/，提交的是 NOTES 中的数据与解释。保留采集工具版本、警告和实际单位；ncu Duration 与普通 Event 不同口径，不能互相相除计算加速比。

Nsight Systems 使用能力已有 Day 10 验收证据，不新增独立系统优化任务。如要复查最终程序的分配/H2D/kernel/D2H/同步位置，可选：

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda --sample=none \
  --output=build/final_softmax_warp_timeline ./build/softmax_final_warp --profile
```

## 7. 最终提交与通过标准

提交三个源文件、Event TODO 实现、构建输出、23 个 shape 的正确性输出及退出码、三版 memcheck、三轮原始 Event 结果、三份 ncu 输出/报告位置，以及完成的 NOTES。最后在本 README 下方用自己的话简要总结实现和优化过程。

- [x] 从空 kernel 独立完成朴素版，不复制旧完整实现；学习者已确认。
- [x] 两版优化展示 shared/reduction 和 warp/shuffle，当前接口与测试范围内边界与同步正确。
- [x] 三版全部正确性通过，CUDA 错误检查有效，memcheck 无错误与泄漏。
- [x] 已补齐 Event 核心步骤，三版同条件 Benchmark 完成复测，计时范围与单位已核对；小 shape 不稳定收益已明确记录。
- [x] 使用 Nsight Compute 解释性能与限制，不把单个指标当作因果证明；会区分 Nsight Systems 的时间线口径。
- [x] README/NOTES 解释每版改变了什么、预期影响、实际证据和未证实原因。

导师已根据上述实际证据完成最终 Review。CUDA Bootcamp 已达到进入 CUDALM 的最低能力门槛。停止继续扩展 Bootcamp；FP16/half2、float4、RMSNorm、fusion 不新增为本次必修版本。

## 最终总结（依据源码、学习者回答与实验归档）

以下是本轮实现与答复的归档摘要，不是学习者原话的逐字引用；原始回答、修订与导师修正见 Notes 第 5 节。

- naive：一个线程处理一行，稳定 Softmax 分别求 max、指数和并写归一化结果。长行的相邻 lane 地址以 hidden 为步长；单线程循环读取相邻元素是时间局部性，不等于同一条 warp 指令的合并访问。
- shared：一个 block 处理一行，线程按 tid 起步、以 block 大小遍历列。局部 max/sum 写 shared，再用连续活跃线程的 tree reduction；无有效列贡献 -INFINITY/0，但仍到达初始化及逐轮 barrier。shared 缓存的是规约中间值，不是整行 input。
- warp：每 warp 用 shuffle 合并寄存器中的局部结果，leader 写 shared partial；block barrier 后由 warp 0 做第二级规约，写行标量后再同步，整个 block 才安全读取。无有效数据的 lane 用单位元参与；full mask 声明参与集合，并不自动激活跳过调用的线程。
- 计时与验证：CPU double Reference、23 个 shape、CUDA 错误检查和 memcheck 通过。Event 在同一 default stream 包围 100 次 launch，计时外分配、传输和暖机 10 次，等待 stop 后求平均；不混用 ncu replay 的 Duration。
- 优化一：主 shape 的 grid 从 1 增到 128，跨 SM 并行度与合并访问同时改善。global-load request 保持 49152，sector 从 1572864 降到 196608，即每 request 从 32 降到 4，不将其解释成 input 读取次数减少。
- 优化二：shared-load/store warp 指令分别从 34816/18432 降到 2304/2304，block barrier 阶段从 18 降到 4；两项成本同时变化，未独立分离各自贡献。
- 同配置复测：在 (128,4096) 上 naive/shared/warp 平均为 1.238534333/0.005207333/0.004801667 ms。shared 相对 naive 加速约 237.844×，warp 相对 shared 延迟降低约 7.79%；(128,1024) 的第二次优化降低约 16.31%。小尺寸收益或幅度不稳定，所有实验组分别保留，不挑选轮次或推广到所有形状。
- 指标与限制：warp 更快，不要求 occupancy、SM 吞吐百分比同时升高，也不能据最大 stall 认定唯一瓶颈。学习者最后答复中的 DRAM 方向由导师核对为 33.09%→35.76%（上升）；SM 为 27.11%→20.23%（下降）。该数值修正不改变“指标不是性能评分”的结论。现有证据不等于完整生产级 Softmax 或所有 workload 的性能保证。
