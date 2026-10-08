# Day 11：Nsight Compute 单 kernel 分析

今天不写新 kernel，复用已完成的 FP32 Softmax V1/V2/V3。你负责采集、填写分析与解释，我负责 Review 和梳理证据。目标不是让 V3 一定获胜，而是解释它为什么赢或输。

当前状态：Day 11 基础验收通过，三段收尾答复与导师修正见 [分析笔记](NOTES.md)。本页保留任务与复现实验步骤；下一任务为 [Day 12 综合复盘](../notes/bootcamp_summary.md)，最终独立 Softmax 验收尚未进行。

## 1. 今天要理解什么

Nsight Systems 看主机提交、搬运、kernel 与 gap 的时间线；Nsight Compute 看某个 kernel 内部的执行与资源使用。今天只关注计划中的六类指标，不扩展到 PTX、极限优化或完整 Roofline 分析。

| 指标 | 今天的含义与注意点 |
| --- | --- |
| Kernel Duration | 目标 kernel 的 GPU 执行时间，记录为 µs；不是 CPU launch API 持续时间。 |
| Memory Throughput | `SpeedOfLight` 中的内存子系统吞吐百分比，涵盖多个层级，不等于显存 GB/s。 |
| DRAM Throughput | DRAM 吞吐相对峰值的百分比；另外记录 `MemoryWorkloadAnalysis` 中以 GB/s 表示的实际 DRAM 字节吞吐。 |
| Compute (SM) Throughput | SM 相关吞吐指标的汇总百分比，不是 FP32 FLOP/s，也不能直接称为“计算单元利用率”。 |
| Occupancy | 驻留的活跃 warp 数相对 SM 最大容量的比例；分别记录 theoretical 和 achieved，不等于有多少 SM 正在工作，也不是越高一定越快。 |
| Warp Stall | warp 暂时不能发出下一条指令的原因；先了解 barrier、long/short scoreboard、wait、math pipe throttle。 |

吞吐汇总指标取其组成指标中最高的相对峰值百分比，因此内存汇总与 DRAM 百分比可能差别很大。Occupancy 与 stall 不能脱离 Duration、吞吐和源码单独判断性能。[NVIDIA 指标与 section 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)。

本机 `WarpStateStats` 图的单位是 `cycles/instruction`：表示该状态的平均 warp 周期数相对发出指令数的比值，不是 kernel 总耗时的百分比。barrier 是等 block 同伴；long scoreboard 通常是在等 L1/TEX 路径上的访存依赖；short scoreboard 可能涉及 shared 等非 L1/TEX 操作的依赖，不能一概等同于 bank conflict；wait 是等待固定延迟执行依赖；math pipe throttle 是所需数学管线暂时不能接收指令。它们提示调查方向，不直接证明哪行代码占用了多少时间。[NVIDIA Warp Stall 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)。

## 2. 固定实验条件

- 固定 `CUDA_VISIBLE_DEVICES=0`，不要混用两张卡的结果；尽量避开同卡其他任务。
- 三版使用相同 FP32 输入、同一构建配置、`block=256`。不修改 kernel、block 或数学函数，避免同时改变多个变量。
- `--profile` 已固定 `shape=(128,4096)`、`grid=128`，只启动一次目标 kernel；它不是正确性测试，也没有 Event warm-up。
- 正常运行使用已有 CUDA Event 框架：`warmup=10`、`iterations=100`，不计 malloc 与 H2D/D2H。
- 新测的 Event 结果只与本轮同条件结果比较，不把 Day 9 历史数据拼进本轮。

当前构建缓存是 `Debug`、`sm_75`，本次准备不改变它。先记录实际配置；如果你主动切换 Release，必须重新编译三版并重做整组对照。`Debug` 名称本身不等于启用了 CUDA 的 `-G` 设备调试。

在仓库根目录运行以下命令；若终端在 `build/`，先返回根目录。

```bash
cmake -S . -B build
cmake --build build --target softmax_v1 softmax_v2 softmax_v3 -j 8
nvidia-smi --query-gpu=index,name,driver_version --format=csv
nvcc --version
ncu --version
rg 'CMAKE_BUILD_TYPE:|CUDA_BOOTCAMP_ARCH:' build/CMakeCache.txt
```

## 3. 先复查正确性，再测三轮 Event

```bash
for v in v1 v2 v3; do
  CUDA_VISIBLE_DEVICES=0 "./build/softmax_${v}" --correctness-only
  check_status=$?
  printf '%s 正确性退出码：%d\n' "$v" "$check_status"
  if [ "$check_status" -ne 0 ]; then break; fi
done
```

确认三版全部 `PASS` 且退出码都是 0 后再继续。如果改了 kernel，还需重新运行 Compute Sanitizer；本节仅复用之前已验收的实现。后面的命令若出现非零退出或失败，也应先停止实验排查。

```bash
# 丢弃一次整程序运行，随后交替测三轮。
for v in v1 v2 v3; do
  CUDA_VISIBLE_DEVICES=0 "./build/softmax_${v}" >/dev/null || break
done
for round in 1 2 3; do
  for v in v1 v2 v3; do
    CUDA_VISIBLE_DEVICES=0 "./build/softmax_${v}" || break
  done
done
```

保存三轮原始输出；先对每版的 `(128,4096)` 求平均延迟与极差。`V2 相对 V1 加速比 = V1 平均延迟 / V2 平均延迟`，`V3 相对 V2 加速比 = V2 平均延迟 / V3 平均延迟`。小幅差异若与运行波动相近，结论应写“收益不稳定”，不要强行解释成加速。

## 4. 亲自采集三份 Nsight Compute 报告

本机 Nsight Compute 2022.3.0 提供以下四个 section；额外四个计数器用于延续之前 shared 规约与 float4 request/sector 的对照。只分析表格里要求的项目，不必解释报告中的所有指标。

```bash
EXTRA_METRICS=l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum

for v in v1 v2 v3; do
  CUDA_VISIBLE_DEVICES=0 ncu \
    --kernel-name "regex:softmax_${v}_kernel" \
    --launch-count 1 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    --metrics "$EXTRA_METRICS" \
    --page raw \
    --export "build/day11_softmax_${v}" \
    "./build/softmax_${v}" --profile || break
done
```

报告为 `build/day11_softmax_v1.ncu-rep` 等三个文件。重跑时请改输出前缀，保留上一组报告；不需要强制覆盖。报告已被 Git 忽略，分析笔记才提交到仓库。

三个命令保持同样的 clock/cache/replay 设置。本机默认 kernel replay、`clock-control=base`、`cache-control=all`；采集可能包含多个 pass，且与暖机后的 Event 不同。不要把 ncu Duration 当作 Event latency 的替代品，也不要将两种口径的数值相除作为加速比。若出现权限错误、`n/a` 或异常时钟警告，先保留完整输出给我，不要先改系统配置。[NVIDIA 采集与 replay 说明](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options)。

无需重新采集即可再次查看已有报告：

```bash
ncu --import build/day11_softmax_v1.ncu-rep --page details
ncu --import build/day11_softmax_v1.ncu-rep --page raw
```

### 从哪里取值

| 记录项 | section 标签或 raw 指标 | 单位 |
| --- | --- | --- |
| Kernel Duration | `gpu__time_duration.sum` | µs；若输出 ns/ms，先换算 |
| Memory Throughput | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | % |
| DRAM Throughput | `SpeedOfLight` 的 `DRAM Throughput` | % |
| 实际 DRAM 字节吞吐 | `dram__bytes.sum.per_second` | GB/s；若输出 byte/s，除以 10⁹ |
| Compute (SM) Throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | % |
| Theoretical Occupancy | `sm__maximum_warps_per_active_cycle_pct` | % |
| Achieved Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | % |
| 主要 stall | `WarpStateStats` 图；raw 中 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` | cycles/instruction |
| global-load request/sector | `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | request / sector |
| shared-load/store 指令 | `smsp__inst_executed_op_shared_ld.sum` / `smsp__inst_executed_op_shared_st.sum` | warp 指令 |

注意：两个 section 都可能显示 `Memory Throughput`，必须带 section 名与单位区分。L1/TEX sector 不等于 DRAM 读取次数或流量；`sector/request` 在标量与 float4 之间不能简单按“越低越好”排名。

本机 2022.3 的 raw 输出将上述 stall ratio 的单位显示为 `inst`，但 section 图轴定义为 `Cycles per Instruction`；记录时按该语义理解，不要误写成百分比。它与先前 Reduction 使用的 `per_warp_active.pct` 不是同一口径。

## 5. 你要回答的问题

1. 同轮 Event 中 V2 比 V1、V3 比 V2 分别快多少？三轮波动是否足以影响判断？ncu 的排序是否一致？
2. V1 → V2 的 shared-load/store 指令和 barrier stall 怎么变化？哪些指标直接支持减少 shared 访问/同步成本，哪些只是推断？
3. V2 → V3 的 global-load request、sector 与 sector/request 怎么变化？request 下降能否说明 DRAM 流量也下降？
4. Memory Throughput 与 DRAM Throughput 是否接近？如果 DRAM 很低，能否直接断言 compute-bound？结合其他指标给出有限结论。
5. theoretical/achieved occupancy 是否不同？如果更快的版本 occupancy 更低，该如何解释，而不是把它判为失败？
6. 每版主要的 1–2 个 stall 是什么？能否联系源码提出原因，同时说明当前数据不能证明什么？

## 6. 如何判断完成，交什么给我 Review

- 三版正确性全部通过，保留退出状态；同配置完成三轮 Event 对照。
- 三份报告对应同样的 `(128,4096)`、`grid=128`、`block=256`，无关键指标缺失。
- 填写 [分析模板](NOTES.md)，数值携带单位，写出 V1、V2、V3、瓶颈判断、V3 为什么赢或输。
- 把原始 Event 输出、三个 ncu 输出和六个问题的回答发给我；区分“测到了什么”和“推测为什么”。

本节通过后才按计划评估综合补缺与最终独立 Softmax 验收。当前框架准备完成不等于 Day 11 或 Bootcamp 已验收。

## 7. 收尾任务：用自己的话解释已有结果（已完成 Review）

以下保留已完成的收尾要求。学习者三段回复已由导师整理进 Notes，并纠正绝对瓶颈表述；不要求继续重跑 Benchmark/Profile 或深入底层 stall 实现。

可以直接回复以下三组问题，每组约 3–5 句话，或填入 Notes 对应小节；不必重复抄完整数据表：

1. **V1 → V2**：同轮 Event 平均延迟如何变化？源码的 shared 访问与 block barrier 如何变化？哪些证据支持优化有效，哪些成本贡献仍不能独立分离？这组对应 Notes 的 V1/V2 小节。
2. **V2 → V3**：request、总 sector 与同轮延迟如何变化？为什么请求更少不保证整体更快？结合主要形状与其他形状，给出有范围限定的结论；若提出具体变慢原因，区分已测事实与待验证推断。这组对应 V3 与“为什么 V3 赢或输”。
3. **瓶颈与 occupancy**：结合 long scoreboard、DRAM 吞吐与 Issue Active 给出合理调查方向，并说明尚不能证明什么。说明等待 warp 是否仍驻留，以及为何 occupancy 高也不保证每周期发出指令。这组对应“瓶颈判断”，同时完成此前两个理解检查的修正。

完成标准：关键数值携带单位；瓶颈解释至少结合两项指标与源码；区分直接证据和推断，不将归一化 stall 当作总耗时比例。不要求证明唯一瓶颈，也不要求 V3 一定加速。

收到回答后由导师 Review 并整理进 Notes，不代写学习者的分析。本节通过后按计划进行综合复盘，再准备最终独立 Softmax 验收；RMSNorm 和 fusion 属于可选补缺，不自动新增为必做任务。
