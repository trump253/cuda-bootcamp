# 最终独立 Softmax 实验记录

状态：2026-10-09 三版实现、正确性、memcheck、Event 与 Profile 已 Review，同条件三轮复测已通过。学习者已提交实现解释初稿；跨 warp 可见性、global request/sector 与单指标判断仍需简短修订，尚未通过最终验收。不使用 Day 9 的旧数据代替本轮实验。

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

公式：`平均=三轮之和/3`；`极差/平均=(最大-最小)/平均`；`加速比=前版平均延迟/后版平均延迟`。以下加速比只是首次交替组的观测值，右列保留当时判断；后续复测结论见第 3.4 节，不能作为跨环境保证。

| shape | shared 相对 naive 加速比 | warp 相对 shared 加速比 | warp 相对 shared 延迟降低 | 导师统计判断 |
| --- | --- | --- | --- | --- |
| (1,128) | 4.550× | 1.054× | 5.08% | 小尺寸波动明显，不能据平均值确认稳定收益 |
| (16,512) | 20.958× | 1.082× | 7.61% | 小尺寸波动明显，不能据平均值确认稳定收益 |
| (128,1024) | 102.189× | 1.195× | 16.34% | shared → warp 三轮方向一致，波动小于收益；naive 首轮偏慢，精确倍数待复测 |
| (128,4096) | 255.726× | 1.102× | 9.27% | shared → warp 三轮方向一致，波动小于收益；naive 首轮偏慢，精确倍数待复测 |

naive 的独立基线组三轮稳定，但交替对照组第一轮明显偏慢：主 shape 的极差/平均为 21.25%，其他三个 shape 约 36.83%～37.47%。两组都保留，不把独立基线“补入”交替组，也不把第一轮偏慢直接归因为升频或初始化；原日志没有足够证据定位其原因。

shared → warp 在 `(128,1024)` / `(128,4096)` 上的平均延迟分别降低约 16.34% / 9.27%，各版极差/平均低于 1%，且逐轮方向一致。本轮支持这两个 shape 的性能收益；小尺寸平均值虽然也更低，但波动足以影响判断，不要求其必然获益。naive → 优化版的数量级改善明显，精确且稳定的三版加速倍数仍需复测确认。

### 3.3 最小复测命令（学习者已完成）

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

复测原始输出已于 2026-10-09 提交，完整记录见第 3.4 节。现有源码未改变，无需继续复测或重采三份 ncu。

### 3.4 整组暖机后的交替三轮复测（本轮比较基准）

学习者先完整执行 naive/shared/warp 两组整程序暖机，再交替运行三轮，每次仍为 warmup=10、iterations=100。附件为 `/root/.codex/attachments/8423121b-e25e-4ab6-82ed-cdd525129da1/已粘贴的文本.txt`；共 207 条正确性 PASS、0 条 FAIL，即三版各 23 个 shape 重复三次。导师仅解析提交数据，未代跑新实验；三版源码及 Event 支撑与上一轮已 Review 的版本哈希相同。

| 版本 | shape | 第 1 轮（ms） | 第 2 轮（ms） | 第 3 轮（ms） | 平均（ms） | 极差/平均（%） |
| --- | --- | --- | --- | --- | --- | --- |
| naive | (1,128) | 0.010828 | 0.010854 | 0.010836 | 0.010839333 | 0.240 |
| naive | (16,512) | 0.054067 | 0.053731 | 0.053743 | 0.053847000 | 0.624 |
| naive | (128,1024) | 0.311910 | 0.311990 | 0.311377 | 0.311759000 | 0.197 |
| naive | (128,4096) | 1.239814 | 1.237445 | 1.238344 | 1.238534333 | 0.191 |
| shared | (1,128) | 0.002609 | 0.002806 | 0.002847 | 0.002754000 | 8.642 |
| shared | (16,512) | 0.002765 | 0.002764 | 0.002744 | 0.002757667 | 0.762 |
| shared | (128,1024) | 0.003512 | 0.003504 | 0.003491 | 0.003502333 | 0.600 |
| shared | (128,4096) | 0.005197 | 0.005187 | 0.005238 | 0.005207333 | 0.979 |
| warp | (1,128) | 0.002990 | 0.002313 | 0.002867 | 0.002723333 | 24.859 |
| warp | (16,512) | 0.002731 | 0.002396 | 0.002683 | 0.002603333 | 12.868 |
| warp | (128,1024) | 0.002936 | 0.002929 | 0.002928 | 0.002931000 | 0.273 |
| warp | (128,4096) | 0.004904 | 0.004748 | 0.004753 | 0.004801667 | 3.249 |

| shape | shared 相对 naive 加速比 | warp 相对 shared 加速比 | warp 相对 shared 延迟降低 | 本轮判断 |
| --- | --- | --- | --- | --- |
| (1,128) | 3.936× | 1.011× | 1.11% | shared → warp 收益或幅度不稳定，不作普遍保证 |
| (16,512) | 19.526× | 1.059× | 5.60% | shared → warp 收益或幅度不稳定，不作普遍保证 |
| (128,1024) | 89.015× | 1.195× | 16.31% | warp 三轮均更快，本轮支持该形状收益 |
| (128,4096) | 237.844× | 1.084× | 7.79% | warp 三轮均更快，本轮支持该形状收益 |

`(128,4096)` 的 naive 波动从首次交替组的 21.25% 降到本组约 0.191%，shared/warp 约为 0.979%/3.249%。shared → warp 的三轮方向仍一致，平均延迟降低约 7.79%；`(128,1024)` 平均降低约 16.31%。本组主 shape 的 shared/naive 与 warp/naive 加速比分别约 237.844× / 257.938×，仅描述本组配置与输入，不推广到所有 workload。

两次交替组分别保留，不拼接平均值。复测在本轮范围内确认了两次优化及小形状收益不稳定的限制，不要求所有小形状都稳定获益，也不继续追加测量。原始日志未独立记录干扰/频率，不能据复测变稳定反推首次偏慢的确定原因。

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

学习者已在 2026-10-09 对话中提交初稿，下面保留原文，并明确区分导师反馈；不把反馈当作学习者已自行修订。

### 5.1 学习者原始回答

1. 朴素版的warp地址分布均跨步长hidden，无法合并访存。线程内连续访存指的是单个线程在循环中的读取，warp合并访存是固定某次循环中一个warp32个线程访问了连续的地址，触发了合并访存，不是一回事。
2. shared版本从block大小的一半开始规约，保证只有小于步长的tid可以参与规约从而避免越界。两次同步（barrier），第一次为了让block内各个线程均完成自己的任务并顺利写入共享内存，第二次是在规约中每轮保证当前轮次的规约计算完成，从而让依赖本轮结果的下一轮规约计算正确。
3. warp mask控制一个warp内参与同步的线程，让warp的所有线程均参与同步保证结果正确。跨warp通过共享内存共享结果，行max/sum只有在warp写入共享内存row_max/sum_shared才对所有线程可见，寄存器里的值只在当前线程可见。
4. 第一次优化使用共享内存缓存结果，减少了全局内存的访问，且可以合并访存使得sector减少。第二次优化使用warp洗牌减少了共享内存的访问。global-load和shared-load的大幅下降可以支持优化的结论，barrier同步代价可能会拖慢速度，但是只是推断
5. 尺寸越小收益越不稳定，大尺寸基本都有稳定收益。如何避免把 occupancy、吞吐百分比或最大 stall 单独当作结论？这是什么意思？
6. 未填写。

### 5.2 导师逐项反馈

1. 基本正确。当前 naive 的同一次读取指令中，相邻活跃 lane 的地址差是 `hidden*sizeof(float)`；`hidden=4096` 时为 16384 字节。单线程循环读取相邻元素是时间局部性，不等于 warp 合并访存。“无法合并”要限定这里的跨步布局与 shape，不能推广到 `hidden=1` 等情况；访存分析比较同一条指令下活跃 lane 的地址，不要求总有 32 个有效 lane。
2. 两类同步的解释正确，但不是实际只有两次 barrier：max 与 sum 各 1 次初始写入同步、8 次 tree 同步，共 18 个阶段。`tid<stride` 选择本轮合并对的写入者，避免重复合并；当前 256-thread、二次幂 stride 下 `tid+stride` 也在有效范围。没有有效列的线程仍以 max 的 `-INFINITY`、sum 的 `0` 初始化 shared 并到达所有 barrier，这已由源码和边界测试确认。
3. shared 交接方向正确，但“写入就对整个 block 可见”遗漏了顺序保证：各 warp 写 partial → 第一次 block barrier → warp 0 读 partial 并写 row 标量 → 第二次 block barrier → 所有线程读取 row 标量。不能仅因地址属于 shared 就跳过写后同步。mask 声明参与集合，不会强迫未执行该调用的 lane 自动加入；本版 full mask 合法是因为所有指定的真实 lane 都执行相同 shuffle，warp 0 第二级规约的后 24 个 lane 仍以单位元参与。依据：[CUDA 11.8 同步函数](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#synchronization-functions)、[Warp Shuffle 参与条件](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#warp-shuffle-functions)。
4. sector 减少和 shared 指令减少的方向正确，但 global-load request 并没有下降：三版都是 49152。shared 存的是局部 max/sum，不是缓存整行 input；三版仍各有 max/sum/write 三遍 input 读取。naive → shared 的 sector 从 1572864 降到 196608，即每 request 从 32 sector 降到 4，源于线程映射改为同一行内的连续列访问；主 shape 的 grid 还从 1 增到 128。shared → warp 的 shared 指令与 barrier 阶段同时下降，18→4 是源码事实，二者各自贡献没有被独立分离。不能把 L1/TEX sector 当作 DRAM 流量，或把全部加速归给单一 shared/shuffle 指令。
5. “本轮小形状收益不稳定、两个较大形状逐轮有收益”可以成立，不能保证所有更大尺寸都稳定。“避免单指标结论”就是不用这些指标给性能打分：warp 更快，但报告中的 occupancy 从 43.82% 略降到 43.46%、SM Throughput 从 27.11% 降到 20.23%；long scoreboard 从 6.37 升到 6.60 cycles/instruction，也不表示 warp 一定更慢或唯一瓶颈已定位。先看同配置 Event 的延迟与波动，再用源码变化、shared/request/sector、相关吞吐及 Issue Active 等解释可支持的范围。stall 是 warp 等待统计，不是可相加的 kernel 耗时百分比。依据：[NVIDIA Profile section 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)。
6. 不新增作业；沿用已通过的 Day 10 Nsight Systems 时间线证据，见 [Day 10 Notes](../08_nsys/NOTES.md)。本轮空回答不否定此前已验收的工具使用能力。

### 5.3 最后需要学习者确认的内容

只修订第 3、4 点，并用本轮一个反例回答第 5 点，各一两句即可；无需重写六题、重复测试或证明唯一瓶颈。可以直接在对话中提交，导师再按学习者修订归档 README 最终总结。请同时用一句话确认本轮从空框架独立实现、未复制 Day 9 完整 kernel。第 1/2 点、现有 Event 流程与 Day 10 工具能力不重复考核。

学习者最终修订：待提交。

## 6. 验收交付

- 三版源码与 Event 核心步骤：已提交；导师未改写学习者源码。
- 构建、23 个 shape 正确性、memcheck：已通过本轮检查。
- 三轮 Benchmark 与 ncu：已提交并记录；整组暖机后的同条件复测通过，不再追加性能实验，小尺寸不保证收益。
- 学习者回答：已提交初稿，第 1/2 点基本正确，第 3/4/5 点待简短修订；第 6 点沿用 Day 10。README 最终总结在修订后归档，不把导师反馈当作学习者已理解。
- 导师最终 Review：代码和实验部分通过；待上述概念修订及独立实现确认，尚未通过最终验收，不新增 RMSNorm/fusion/FP16 等算子门槛。
