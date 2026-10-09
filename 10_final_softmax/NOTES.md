# 最终独立 Softmax 实验记录

状态：2026-10-09 已提交三版实现、正确性、memcheck、Event 与 Profile。导师已核对源码和数据；同条件 Benchmark 复测与学习者自己的优化总结待完成，尚未通过最终验收。不使用 Day 9 的旧数据代替本轮实验。

## 1. 环境与实现

- 采集命令固定 `CUDA_VISIBLE_DEVICES=0`，报告设备为 RTX 2080 Ti，68 SM。同卡其他任务、逐次频率及温度未随原始日志记录，不能假定完全无干扰。
- Review 时本地环境：CUDA 11.8 / nvcc 11.8.89、驱动 570.172.08、CMake 3.16.3、ncu 2022.3.0.0、nsys 2022.4.2.50；这不是每次采集时的独立环境快照。
- 三版现有编译命令均为 Debug / `-arch=sm_75`，含主机 `-g`，无设备 `-G` 或 `--use_fast_math`。本轮未切换 Release。
- 输入输出为 FP32 行主序，Reference 使用相同 FP32 输入以 double 计算；三版均使用 `fmaxf` / `expf`，没有用快速指数函数替换来混入另一变量。
- naive：一个线程处理一行，block=128，grid=ceil(rows/128)。shared/warp：一个 block 处理一行，block=256，grid=rows；warp 版每 block 有 8 个 warp。
- Event：分配、H2D、正确性检查及 warmup=10 在计时外；default stream 上 start → 100 次 launch → stop，等待 stop 完成后取 elapsed_ms/100。各版 launch wrapper 均检查 CUDA 错误。Event 区间可能包含 GPU 等待后续主机提交的空隙，不是独立分解出的纯算术时间。

## 2. 正确性与内存安全

下列误差为本次提交中该版本所有正确性输出的最大诊断值，均为无量纲；重复运行不增加独立 shape 数。原日志共 463 条 PASS、0 条 FAIL，每版覆盖 23 个独立 shape。

| 版本 | 23 个 shape | 最大绝对误差 | 最大相对误差 | 最大行和误差 | 程序退出码 / memcheck |
| --- | --- | --- | --- | --- | --- |
| naive | 全部 PASS | 8.55940e-8 | 6.07036e-6 | 5.94367e-6 | 原日志显式退出码 0；0 errors、0 bytes leaked，memcheck 显式退出码 0 |
| shared | 全部 PASS | 4.53892e-8 | 2.74287e-7 | 1.44107e-7 | 原日志未单列退出码；set -e 对照循环继续；0 errors、0 bytes leaked |
| warp | 全部 PASS | 4.29456e-8 | 2.67147e-7 | 1.55589e-7 | 原日志未单列退出码；set -e 对照循环继续；0 errors、0 bytes leaked |

导师在本次 Review 中重新构建三目标，并仅运行三版 `--correctness-only`：各 23 项 PASS、退出码均为 0。memcheck、Event 和 Profile 使用学习者提交的输出，没有代跑新一轮性能实验。现有有限输入、正维度及非重叠指针约定下，未发现阻塞性的索引、尾部单位元或同步错误。

原始提交：2026-10-09 对话附件，本地位置为 `/root/.codex/attachments/a4fefd94-4bce-4711-a588-cfeb166ad133/已粘贴的文本.txt`。这个位置不是 Git 中的共享文件；关键数值逐项整理如下。

## 3. 三轮 Event：分开保留两组实验

### 3.1 naive 独立基线组三轮

这是三版交替对照之前的独立三轮，均为 warmup=10、iterations=100。不与后面的 shared/warp 测量拼成另一组“同条件三版对照”。

| shape | 第 1 轮（ms） | 第 2 轮（ms） | 第 3 轮（ms） | 平均（ms） | 极差/平均（%） |
| --- | --- | --- | --- | --- | --- |
| (1,128) | 0.010804 | 0.010836 | 0.010859 | 0.010833000 | 0.508 |
| (16,512) | 0.054067 | 0.054008 | 0.054088 | 0.054054333 | 0.148 |
| (128,1024) | 0.312442 | 0.311893 | 0.312499 | 0.312278000 | 0.194 |
| (128,4096) | 1.243031 | 1.241932 | 1.242093 | 1.242352000 | 0.088 |

### 3.2 三版交替对照组三轮

每轮按 naive → shared → warp 顺序运行，均为 warmup=10、iterations=100。以下完整保留三轮，不删除交替组较慢的第一轮。

| 版本 | shape | 第 1 轮（ms） | 第 2 轮（ms） | 第 3 轮（ms） | 平均（ms） | 极差/平均（%） |
| --- | --- | --- | --- | --- | --- | --- |
| naive | (1,128) | 0.015435 | 0.010871 | 0.010875 | 0.012393667 | 36.825 |
| naive | (16,512) | 0.076786 | 0.053969 | 0.053742 | 0.061499000 | 37.471 |
| naive | (128,1024) | 0.443213 | 0.312054 | 0.312506 | 0.355924333 | 36.850 |
| naive | (128,4096) | 1.522018 | 1.240240 | 1.238577 | 1.333611667 | 21.254 |
| shared | (1,128) | 0.003111 | 0.002621 | 0.002439 | 0.002723667 | 24.673 |
| shared | (16,512) | 0.003126 | 0.002825 | 0.002852 | 0.002934333 | 10.258 |
| shared | (128,1024) | 0.003482 | 0.003485 | 0.003482 | 0.003483000 | 0.086 |
| shared | (128,4096) | 0.005238 | 0.005219 | 0.005188 | 0.005215000 | 0.959 |
| warp | (1,128) | 0.002492 | 0.002847 | 0.002417 | 0.002585333 | 16.632 |
| warp | (16,512) | 0.002873 | 0.002865 | 0.002395 | 0.002711000 | 17.632 |
| warp | (128,1024) | 0.002908 | 0.002907 | 0.002927 | 0.002914000 | 0.686 |
| warp | (128,4096) | 0.004753 | 0.004710 | 0.004731 | 0.004731333 | 0.909 |

公式：`平均=三轮之和/3`；`极差/平均=(最大-最小)/平均`；`加速比=前版平均延迟/后版平均延迟`。以下加速比只是本组观测值，稳定性须结合右列，不能作为跨环境保证。

| shape | shared 相对 naive 加速比 | warp 相对 shared 加速比 | warp 相对 shared 延迟降低 | 导师统计判断 |
| --- | --- | --- | --- | --- |
| (1,128) | 4.550× | 1.054× | 5.08% | 小尺寸波动明显，不能据平均值确认稳定收益 |
| (16,512) | 20.958× | 1.082× | 7.61% | 小尺寸波动明显，不能据平均值确认稳定收益 |
| (128,1024) | 102.189× | 1.195× | 16.34% | shared → warp 三轮方向一致，波动小于收益；naive 首轮偏慢，精确倍数待复测 |
| (128,4096) | 255.726× | 1.102× | 9.27% | shared → warp 三轮方向一致，波动小于收益；naive 首轮偏慢，精确倍数待复测 |

naive 的独立基线组三轮稳定，但交替对照组第一轮明显偏慢：主 shape 的极差/平均为 21.25%，其他三个 shape 约 36.83%～37.47%。两组都保留，不把独立基线“补入”交替组，也不把第一轮偏慢直接归因为升频或初始化；原日志没有足够证据定位其原因。

shared → warp 在 `(128,1024)` / `(128,4096)` 上的平均延迟分别降低约 16.34% / 9.27%，各版极差/平均低于 1%，且逐轮方向一致。本轮支持这两个 shape 的性能收益；小尺寸平均值虽然也更低，但波动足以影响判断，不要求其必然获益。naive → 优化版的数量级改善明显，精确且稳定的三版加速倍数仍需复测确认。

### 3.3 最小复测命令（学习者执行）

在仓库根目录一次执行整个命令块，减少手动逐条启动之间的停顿；不在 Sanitizer/ncu 下测 Event，不重新混入旧数据。若仍波动，保留结果并限定结论，不无限追求每个小 shape 都稳定，也不新增优化版本。

```bash
(
  set -e
  for warm_group in 1 2; do
    for v in naive shared warp; do
      CUDA_VISIBLE_DEVICES=0 "./build/softmax_final_${v}" >/dev/null
    done
  done
  for round in 1 2 3; do
    for v in naive shared warp; do
      CUDA_VISIBLE_DEVICES=0 "./build/softmax_final_${v}"
    done
  done
)
```

复测原始输出与结论：待提交。现有源码不变时无需重采三份 ncu。

## 4. Nsight Compute 对照

学习者亲自采集，固定 shape=`(128,4096)`，每版只选择一个目标 kernel、14 passes。报告位于 `build/final_softmax_naive.ncu-rep`、`build/final_softmax_shared.ncu-rep`、`build/final_softmax_warp.ncu-rep`；build/ 被 Git 忽略，报告未上传到仓库。导师只读导入三份已有报告复核，下表与原始输出一致，提交输出未见 WARNING。

| 指标 | naive | shared | warp |
| --- | --- | --- | --- |
| GPU Duration（µs） | 约 1730（原始为 1.73 ms） | 10.91 | 10.27 |
| SpeedOfLight：Memory Throughput（%） | 1.32 | 33.09 | 35.76 |
| SpeedOfLight：DRAM Throughput（%） | 0.19 | 33.09 | 35.76 |
| SpeedOfLight：Compute/SM Throughput（%） | 0.09 | 27.11 | 20.23 |
| MemoryWorkloadAnalysis：Memory Throughput（GB/s） | 1.21 | 192.29 | 204.20 |
| Theoretical Occupancy（%） | 100 | 100 | 100 |
| Achieved Occupancy（%） | 12.50 | 43.82 | 43.46 |
| 平均驻留 warp（warp/SM，报告活跃周期口径） | 4.00 | 14.02 | 13.91 |
| Issue Active（每活跃周期比值，无量纲） | 0.05 | 0.24 | 0.23 |
| grid / block（block 数 / thread 数） | 1 / 128 | 128 / 256 | 128 / 256 |
| register/thread（个） | 42 | 42 | 41 |
| 静态 shared/block（字节，按源码精确值） | 0 | 2048 | 72 |
| long scoreboard（cycles/instruction） | 15.78 | 6.37 | 6.60 |
| barrier（cycles/instruction） | 0 | 1.32 | 1.11 |
| short scoreboard（cycles/instruction） | 0.51 | 1.44 | 0.97 |
| wait（cycles/instruction） | 1.86 | 2.15 | 2.07 |
| global-load request（次） | 49152 | 49152 | 49152 |
| global-load sector（个，32 字节/sector） | 1572864 | 196608 | 196608 |
| global-load sector/request（个/次） | 32 | 4 | 4 |
| shared-load warp instructions（条） | 0 | 34816 | 2304 |
| shared-store warp instructions（条） | 0 | 18432 | 2304 |

口径提醒：WarpStateStats 图中的 stall 为每发出指令的平均等待周期，原始 ratio 行可能以 `inst` 显示单位，不能读成 kernel 耗时百分比；occupancy 描述驻留 warp，不是全卡 GPU 利用率。ncu replay 的时间与普通 Event 不同口径，不互相相除。参见 [NVIDIA Profile section 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)。

导师核对的直接变化：

- naive → shared：global-load request 不变，sector 降低 87.5%；同时行映射改变、主 shape 的 grid 从 1 变成 128。不能把加速全部归结为“shared 比 global 快”，这两版的并行度与合并访问也同时改变。
- shared → warp：global-load request/sector 不变；shared-load/store warp instructions 分别下降 93.38% / 87.50%。源码中的 block barrier 阶段从 max/sum 各 9 次、共 18 次变为各 2 次、共 4 次。该对照未独立分离 shared 指令与 barrier 减少各自贡献。
- warp 的 Event 在两个较大 shape 更快，但 achieved occupancy、SM Throughput 不比 shared 更高；这些百分比不是性能评分。long scoreboard 与 DRAM 等指标应结合工作量分析，汇总报告未证明唯一瓶颈。

这些是导师的数据整理与 Review，不代替学习者下面的实现解释。

## 5. 用自己的实现回答

可合并为三段短回答，不需新增代码或追查唯一瓶颈；第 6 题可直接引用已验收的 Day 10 证据。

1. 朴素版的 warp 地址分布是什么？线程内连续读取和 warp 合并访问是否是一回事？
2. shared 版如何规约、处理无有效列的线程？每处 barrier 保护什么？
3. warp 版如何选择 mask？跨 warp 结果如何交接，行 max/sum 何时对其他线程可见？
4. 两次优化各想减少什么成本？哪些变化是 Event/Profile 直接观测，哪些只是推断？
5. 哪些 shape 赢、输或收益不稳定？如何避免把 occupancy、吞吐百分比或最大 stall 单独当作结论？
6. 若用 Nsight Systems 复查，能否指出主机 API、H2D/D2H、GPU kernel 与同步？可引用已验收的 Day 10 证据，无需新增系统优化。

学习者回答：待填写，可直接在对话中提交。请同时说明本轮是否从空框架独立实现，没有复制 Day 9 的完整 kernel。

## 6. 验收交付

- 三版源码与 Event 核心步骤：已提交；导师未改写学习者源码。
- 构建、23 个 shape 正确性、memcheck：已通过本轮检查。
- 三轮 Benchmark 与 ncu：已提交并记录；交替组的 naive 首轮偏慢，同条件复测待提交，小尺寸不保证收益。
- README 的最终实现/优化总结及本页学习者回答：待填写。
- 导师最终 Review：代码和现有证据已核对，待上述收尾；尚未通过最终验收，不新增 RMSNorm/fusion/FP16 等算子门槛。
