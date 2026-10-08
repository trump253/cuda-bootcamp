# 学习进度

## 当前阶段

Day 10 — 容器内 Nsight Systems GUI 与 noVNC 已启动并验证；三份正式报告的完整采集与解释尚未完成，最终独立 Softmax 验收尚未进行

## 已完成

- 已阅读 CUDA Bootcamp 学习计划并确认学习范围。
- 已检查仓库的初始文件。
- 已创建最小 CMake 目标、项目说明、进度记录和 Day 1 练习框架。
- 已检查当前终端的 CUDA 构建与运行环境。
- 已将面向学习者的项目文档和代码注释统一为中文。
- 已完成 Day 1 主体逻辑和 CPU/GPU 独立结果验证逻辑。
- 已使用 CUDA 11.8 成功编译 `vector_add`。
- 已在可用 GPU 环境中运行全部规定规模；CPU/GPU 最大绝对误差均为 0。
- 已正确解释 `N = 1000`、`block = 256` 时的线程映射和尾部边界保护。
- Compute Sanitizer 检查通过：无非法内存访问，CUDA 内存泄漏为 0 字节，
  错误总数为 0。
- Day 1 验收通过。
- 已将 Day 1 归档到 `01_vector_add/`，并将 CUDA 错误检查提取到 `common/`。
- 已创建可编译的 Day 2 Matrix Add 任务框架和中文任务说明。
- `vector_add` 与 `matrix_add` 两个 CMake target 均已编译成功。
- 已完成 Day 2 Matrix Add 的初版二维 kernel、CPU Reference 和四组结果比较；
  当前四组测试均打印 `PASS`。
- 已修复 Day 2 验证返回值，并补齐最大误差、block、grid 和 NaN/Inf 检查。
- Day 2 Matrix Add 正确性验收通过：全部误差为 0，程序退出码为 0，
  Compute Sanitizer 报告 0 字节泄漏和 0 个错误。
- 已创建 Day 2 CUDA Event Benchmark 模板、测试尺寸和中文 TODO。
- 已完成 Day 2 CUDA Event Benchmark 初版；Event 创建、记录、同步、计时和销毁
  流程已基本实现。
- 已修复 Benchmark 的 grid、输出参数和 GB/s 单位换算，并连续运行三次。
- Day 2 Benchmark 验收通过：较大矩阵约达到 500–556 GB/s；三轮结果稳定，
  `4096 × 4096` latency 极差约为 0.022%。
- 已清理 Day 2 完成项的 TODO，补齐 warm-up launch error 检查并记录实验结论。
- Day 2 验收完成。
- 已创建 Day 3–4 Naive Matrix Transpose 的正确性模板和中文任务说明。
- Day 1/Day 2 的 CMake target 已归档为注释，默认构建只保留当前练习。
- 已确认 RTX 2080 Ti 单卡理论显存带宽为 616 GB/s；Matrix Add 实测峰值
  555.82 GB/s，对应约 90.2% 理论带宽利用率。
- 已完成 Naive Matrix Transpose 的 CPU/GPU 初版；所有规定尺寸误差为 0，
  Compute Sanitizer 报告 0 字节泄漏和 0 个错误。
- 已修正 Transpose 输出 shape，并正确解释 V0 的合并读取与跨步写入。
- Naive Matrix Transpose V0 正确性阶段验收通过。
- 已创建 V0 CUDA Event Benchmark 模板和测试尺寸。
- Naive Transpose V0 Benchmark 已通过：三轮结果稳定，`4096 × 4096` 平均
  latency 约 0.324170 ms、有效带宽约 414.03 GB/s。
- 已记录 V0 合并读取、跨步写入和 cache 对有效带宽解释的限制。
- 已归档 Naive Transpose V0 target，并创建 V1 未 padding 的 Shared Memory
  Tiled Transpose 框架、测试尺寸和中文任务说明。
- 已将 V0 验证过的 CPU Reference、误差检查、测试夹具和 CUDA Event Benchmark
  复用到 V1 框架；当前只需实现 tiled kernel。
- V1 初版通过全部正确性与 Compute Sanitizer 检查，但性能低于 V0：
  `4096 × 4096` 平均约 0.47244 ms、284.09 GB/s。
- 已定位 V1 初版性能回退原因：shared tile 写入与读取使用相同转置索引，未完成
  tile 内转置；global output 仍以 `rows` 为步长写入。
- 已完成两种输出线程映射对照：直接让原线程写回时 `4096 × 4096` 平均约
  0.32903 ms；交换线程在输出 tile 中的职责后平均约 0.28031 ms。
- 当前 V1 已实现合并 global load/store；相对 V0 的 0.32417 ms，`4096 × 4096`
  speedup 约为 1.16×，有效带宽约为 478.81 GB/s。
- 当前 V1 Compute Sanitizer 通过：0 errors、0 bytes leaked；未 padding V1 验收完成。
- 已清理 V1 完成项的 TODO，并加入 `kTilePadding` 开关用于 padding 对照实验。
- 已完成 `kTilePadding=1` 的全部正确性测试与 Compute Sanitizer 检查：
  0 errors、0 bytes leaked。
- Padding 版三轮 Benchmark 已完成。`4096 × 4096` 平均 latency 约
  0.259670 ms、有效带宽约 516.88 GB/s；相对未 padding 版的
  0.280313 ms 加速约 1.080×，latency 降低约 7.36%。
- Padding 版相对 Naive Transpose 的 0.324170 ms 加速约 1.248×。
- 已完成转置读取阶段的 bank 映射推导：未 padding 时 warp 集中访问
  bank 0、1、16、17，形成 8-way bank conflict；padding=1 后仅 bank 0
  存在一次 2-way conflict。还需注意 padding=1 的共享内存写入阶段同样会在
  bank 0 产生一次 2-way conflict，因为 `16 × 16` block 的一个 warp 跨两行。
- 已创建 Nsight Compute 单变量对照框架：从同一源文件分别编译 padding=0/1，
  `--profile` 模式只启动一次 `4096 × 4096` kernel；两个目标均编译通过。
- 已完成 Nsight Compute bank-conflict 对照。`4096 × 4096` 共执行 524288 个
  warp：padding=0 的 shared load conflict 为 3670016，即每 warp 7；shared
  store conflict 为 0。padding=1 的 load/store conflict 均为 524288，即各自
  每 warp 1。
- Padding 使 shared load conflict 降低约 85.71%，并引入每 warp 1 个 store
  conflict；load+store 总 conflict 从每 warp 7 降至 2，降低约 71.43%。
- Matrix Transpose 已完成 Reference、Naive CUDA、Correctness、Benchmark、
  Profile、Optimization、Re-benchmark 和 Notes 全部闭环，Day 3–4 验收完成。
- 已将 Matrix Transpose 的 V0、无 padding V1、padding V1、有效带宽、speedup
  和 Nsight Compute bank-conflict 证据整理到 `03_transpose/README.md`。
- 已归档 Matrix Transpose 默认构建目标，并创建可编译的 Sum Reduction V0
  框架；CPU Reference、正确性测试和 CUDA Event Benchmark 均已预置。
- Reduction V0 串行 kernel 已完成；全部正确性尺寸误差为 0，Compute Sanitizer
  报告 0 errors、0 bytes leaked。
- V0 三轮 Benchmark 中 `N=2^20` 稳定在约 19.7816 ms、0.212 GB/s；较小三个
  尺寸的前两轮与第三轮存在约 32% 的运行间波动，推测与 GPU 冷启动及升频有关，
  尚需一次暖机后的复测确认稳定基线。
- 已将性能输出格式统一为数值后显式携带单位（如 `ms`、`GB/s`），并将其记录为
  后续所有模板的固定规范。
- 已在丢弃一次整程序暖机后完成 V0 三轮复测。`N=2^20` 平均 latency 为
  19.781326 ms、有效带宽约 0.212 GB/s，三轮 latency 极差仅约 0.008%；
  `N=2^14` 以上各尺寸极差均低于 0.04%。
- 已正确解释 V0：多个线程直接写同一标量会产生数据竞争；一个 block 中只有
  warp 0 的 lane 0 继续串行循环，其余 warp/lane 退出；只有一个 SM 获得工作，
  无法提供合并访问、延迟隐藏或跨 SM 并行性。Reduction V0 验收完成。
- 已归档 V0 默认构建目标，并创建 V1 Shared Memory Tree Reduction 框架。
  CPU Reference、多轮 GPU reduction、ping-pong partial buffers、正确性验证和
  CUDA Event Benchmark 已预置，学习者只需实现 block 内 tree reduction kernel。
- V1 交错寻址 Shared Memory Tree Reduction 已完成；全部正确性尺寸误差为 0，
  Compute Sanitizer 报告 0 errors、0 bytes leaked。
- V1 暖机后三轮 Benchmark 稳定：`N=2^20` 平均约 0.048428 ms、86.61 GB/s，
  相对 V0 加速约 408.47×；`N=2^24` 平均约 0.639414 ms、104.95 GB/s。
- 已正确解释多轮 reduction、越界线程必须写 0 并到达 barrier，以及交错寻址的
  warp divergence。需注意 barrier 位于非收敛控制流中属于未定义行为，不保证
  一定表现为死锁；每轮 barrier 的精确作用是保证本轮 shared 写入对下一轮可见。
- Reduction V1 正确性、Benchmark、Compute Sanitizer 和概念验收完成。
- 已创建 `CUDA_复习与面试问答.md`，仅整理当前阶段实际遇到的问题、答案与实测
  证据；后续完成新知识点时持续维护，不提前扩展课程范围。
- 已创建 V2 连续活跃线程 Tree Reduction 框架。V1/V2 保持相同输入加载、多轮
  调度、测试规模、warm-up、iterations 和带宽公式，唯一变量为 block 内寻址方式。
- V2 已完成全部正确性测试，最大绝对误差为 0；Compute Sanitizer 报告
  0 errors、0 bytes leaked。
- V2 三轮 Benchmark 已完成：`N=2^20` 平均 0.033143 ms、126.55 GB/s，
  相对 V1 加速约 1.461×；`N=2^24` 平均 0.432722 ms、155.09 GB/s，
  相对 V1 加速约 1.478×。减少 divergence 的优化假设得到性能数据支持。
- 已正确推导 V2 warp 活跃情况：stride=128/64/32 时分别有 4/2/1 个完整
  warp 活跃；从 stride=16 开始 warp 0 内出现分歧。需将 stride=16 的活跃范围
  精确写为 thread 0～15，而不是 0～16。
- 已为 V1/V2 增加 `--profile` 单 kernel 模式。学习者已亲自复现 `N=2^24`
  第一阶段的 Nsight Compute 对照，并提交完整原始输出。
- 学习者已正确区分直接证据与推断：branch 指标没有证明硬件 divergent branch
  减少；直接证据是 V2 warp instructions 和 predicated-on/off thread
  instructions 显著下降，且两版均无 shared-memory bank conflict。需进一步注意，
  branch target 指标不统计被 predication 屏蔽的 lane，不能据此断言两版线程利用率
  或“发散程度完全相同”。V1/V2 Profile 验收完成。
- 已归档 V1 默认构建目标，并创建 V3 Warp Shuffle Reduction 框架。V2/V3
  保持相同测试规模、多轮调度和 Benchmark；V3 核心的 warp shuffle 与两级
  block reduction 留作学习者实现。
- V3 Warp Shuffle Reduction 已由学习者完成。全部正确性测试最大绝对误差为 0，
  程序退出码为 0；Compute Sanitizer 报告 0 errors、0 bytes leaked。
- V3 暖机后三轮 Benchmark 中，`N=2^20` 平均 0.019637 ms、213.59 GB/s，
  相对 V2 加速约 1.688×；`N=2^24` 平均 0.247308 ms、271.36 GB/s，
  相对 V2 加速约 1.750×。Benchmark 已证明 V3 整体更快。
- 已正确解释 V3 的两级归约与 warp 间 barrier。需注意 full mask 不会激活已退出
  lane；当前安全是因为 32 个真实 lane 都执行 shuffle，尾部输入只被置为 0。
  `__shfl_down_sync` 也不是通用内存栅栏，不能替代跨 warp 的 block barrier。
- 学习者已亲自完成 V2/V3 Nsight Compute 对照。V3 相对 V2：总 warp instructions
  减少 66.63%，shared load/store instructions 分别减少 99.22%/88.89%，barrier
  stall 比例从 22.41% 降至 6.39%，累计等待 barrier 的 warp 数减少 87.25%。
  Profile 与 Benchmark 共同支持 V3 优化，Reduction V3 性能闭环验收完成。
- 已完成 Reduction 阶段总结：小规模输入受 GPU 利用率不足与固定 launch 开销影响，
  优化收益容易被掩盖；总归约工作量为 `O(N)`，`O(log N)` 是并行关键路径深度。
  Sum Reduction 算术强度低、通常偏 memory-bound，但当前实现仍同时受到同步、归约
  依赖和指令开销影响。Day 5–6 Reduction 与 Warp 阶段验收完成。
- 已创建 Day 7 GEMM 工程结构。公共框架已提供 CPU Reference、确定性输入、绝对与
  相对误差验证、CUDA Event Benchmark、GFLOP/s 和单 kernel Profile 模式。
- 已创建 V1 Naive GEMM kernel TODO；当前只需完成二维输出映射、边界检查和 K 维
  点积。V2 Tiled 文件已预留 shared tile 接口，但在 V1 验收前不启用构建。
- V1 Naive GEMM kernel 已完成；全部正确性测试通过，退出码为 0，Compute
  Sanitizer 报告 0 errors、0 bytes leaked。
- V1 暖机后三轮 Benchmark 稳定：`256³/512³/1024³` 平均分别为
  795.67/878.97/914.59 GFLOP/s；`1024³` 平均 latency 为 2.348016 ms，三轮
  latency 极差仅约 0.019%。
- 已修正 GEMM grid 与访存理解：grid.x 覆盖 N、grid.y 覆盖 M；固定 inner 时，
  A 在当前 16-thread 半 warp 内是同地址广播，B 才是连续列合并读取。单线程随 K
  的地址变化不能直接用于判断 warp coalescing。
- 已记录单线程在相邻 K 轮读取相邻 A 元素属于时间局部性，不能称作 warp 级合并
  访存；coalescing 应观察同一 warp 在同一条 load 指令中的地址分布。
- 学习者已完成 V1 Naive GEMM 最小 Nsight Compute Profile：`1024³` 单 kernel
  2.35 ms，SM throughput 62.50%，DRAM throughput 0.90%，L1/TEX global-load
  request 67,108,864、sector 134,216,439（约 2.00 sector/request）。Profile
  模式使用全 0 A/B；这些计数不直接等于 DRAM 流量，也不能单独判断唯一瓶颈。
  V1 的 Reference、正确性、Benchmark、Profile 和 Notes 基线验收完成。
- 已启用 V2 Shared Memory Tiled GEMM 构建目标，并补全中文任务、边界与同步
  验收要求；核心 kernel TODO 由学习者完成。
- V2 Tiled GEMM 核心 kernel 已由学习者完成；五组正确性测试通过，退出码为 0，
  Compute Sanitizer 报告 0 errors、0 bytes leaked。两次 block barrier 分别保护
  tile 加载完成和上一阶段读完；尾部补 0 避免无效输入及旧 shared 值参与点积。
- V2 暖机后三轮 Benchmark：`256³/512³/1024³` 平均 latency 分别为
  0.029995/0.200406/1.505280 ms，平均性能为 1118.69/1339.46/1426.64 GFLOP/s；
  相对 V1 分别加速约 1.406×/1.524×/1.560×。性能收益已由 Benchmark 证明。
- 已回答 `threadIdx` 写法问题：直接使用与先存为局部 `tx/ty` 都正确；后者主要
  改善可读性，当前没有证据表明单靠这种写法能改善性能。
- 学习者已完成 V2 `1024³` 单 kernel Nsight Compute 对照：L1/TEX global-load
  request 从 V1 的 67,108,864 降到 4,194,304（减少 93.75%，即 16×），
  sector 从 134,216,439 降到 16,656,832（减少约 87.59%，约 8.06×）。
  GPU duration 从约 2.35 ms 降到 1.51 ms；SM throughput 为 62.50%→73.23%，
  DRAM throughput 为 0.90%→1.41%。请求和 sector 位于 L1/TEX 路径，不能
  直接当作 DRAM 字节数。V2 GEMM 完整性能闭环验收完成。
- 已接入 cuBLAS SGEMM 对照程序，复用 Day 7 的输入、CPU Reference、误差判据
  和 CUDA Event 计时。针对 row-major 矩阵交换 cuBLAS 的 A/B 操作数，包含非方阵
  正确性测试；`gemm_cublas` 编译通过后由学习者完成运行。
- 学习者已完成 cuBLAS SGEMM 对照：五组正确性测试全部 PASS，退出码 0，
  Compute Sanitizer 报告 0 errors、0 bytes leaked。同 GPU 交替三轮 `1024³`
  Benchmark，手写 V2 平均 1.504731 ms、1427.15 GFLOP/s，cuBLAS 平均
  0.222186 ms、9665.25 GFLOP/s；cuBLAS 相对 V2 约快 6.772×。
- 学习者已正确解释本次 6.772× 速度比的适用范围，以及 row-major
  `C=A×B` 对应 column-major `Cᵀ=Bᵀ×Aᵀ` 的操作数交换。Day 7 Naive/Tiled
  GEMM、cuBLAS 对照和 Notes 全部验收完成；按计划停止 GEMM 扩展。
- 已创建 Day 8 FP32/FP16/half2 Vector Add 框架：FP32 对照组和共用 CPU Reference、
  奇数长度正确性验证、CUDA Event Benchmark、单 kernel Profile 入口已准备好。
  FP16 标量和 half2 的核心 kernel 均保留 TODO，等待学习者自行实现。
- 已归档 Day 7 的默认构建目标，当前 CMake 只构建 Day 8 三个版本。
- Day 8 三个 CMake 目标均已用 CUDA 11.8 编译成功；FP32 对照组的全部正确性
  用例通过，CUDA Event Benchmark 路径可运行。未实现的两个 kernel 经检查会显示
  `FAIL` 并返回退出码 1，NaN 哨兵能够识别漏写，尚不能据此视为 Day 8 通过。
- 已解答 GEMM 与简单 element-wise 算子的主性能指标区别，并把 FLOP/s、
  有效带宽、算术强度和“主指标不能单独证明瓶颈”的边界记入复习问答。
- 已澄清 GFLOP/s 与 GB/s 对同一 kernel 可以同时计算，二者在一致的算法
  计数口径下由算术强度联系；强调派生的算法指标不同于硬件层实测吞吐。
- 学习者已填写 Day 8 的 FP16 标量和 half2 kernel。当前 `fp16` 可编译；
  `half2` 因构建默认目标为 `sm_52`，`__hadd2` 在该目标下不可见而编译失败。
  已用 `-arch=sm_75` 对现有 half2 源文件单独编译成功；尚未运行正确性测试。
- 已在 CMake 3.16 中为 Day 8 三个目标统一设置 `-arch=sm_75`，并让 VS Code
  C/C++ 扩展读取 CMake 的 `compile_commands.json`。重新配置后 Debug 模式
  `all` 构建通过，三个目标的编译命令均包含 `-arch=sm_75`；尚未验收运行结果。
- 已按学习者要求把架构选项从 Day 8 三个目标提升为目录级 CUDA 编译选项；
  当前目标及今后在本项目 CMake 中重新启用的历史 CUDA 目标均继承 `sm_75`。
  再次配置与构建通过，三个当前目标各只有一个 `-arch=sm_75`。
- 学习者已完成 Day 8 FP16 标量和 half2 的全部 11 组正确性测试，最大绝对
  误差为 0；两版 Compute Sanitizer 均为 0 errors、0 bytes leaked。
- 学习者交替完成三轮 Debug 构建下的 FP32/FP16/half2 Benchmark，并亲自采集
  `N=2^24` 的 Nsight Compute 对照。half2 相对 FP16 标量平均 latency
  `0.198085 → 0.182479 ms`，同尺寸加速约 `1.086×`；global load/store
  request 各减半，sector 总数不变。已记录在 `06_fp16_half2/README.md`。
- 已核对当前及历史练习的编译参数：当前 Debug 目标使用 nvcc `-g`，没有
  使用会关闭设备优化的 `-G`；旧练习保留的 flags.make 也显示 `-g`、无 `-G`。
  因此撤回“必须 Release 复测才能验收”的前置条件。本节同配置下的正确性、
  Benchmark、Profile、half2 优化对照和 Notes 均已完成，Day 8 验收通过。
- 已创建 Day 9 Softmax V0 框架：逐行串行 kernel 留有中文 TODO；复用的主机
  测试框架已包含稳定的 double CPU Reference、计划规定的 12 个 shape 和边界
  用例、数值与行和验证、CUDA Event Benchmark、单次 Profile 入口。
- 已将 Day 8 的默认构建目标归档为注释，启用 `softmax_v0`；当前 V0 模板
  编译通过。空 kernel 的正确性测试按预期全部 FAIL 并以退出码 1 返回，
  不是 Day 9 正确性验收结果。
- 已为本工程准备本地 Git 仓库；构建目录、独立 CMake 临时目录及 Nsight
  报告文件由 `.gitignore` 排除。首次本地提交已完成。
- 学习者已在 `07_softmax/v0.cu` 填写一线程一行的 V0 算法，包括求最大值、
  指数求和与归一化。17 个正确性用例全部 PASS，程序退出码为 0；Compute
  Sanitizer 报告 0 errors、0 bytes leaked，退出码为 0。
- 学习者完成 V0 三轮 CUDA Event Benchmark：`(128,4096)` 平均
  `1.250821 ms`。Nsight Compute 单次 Profile 为 `1.77 ms`，L1/TEX
  global-load request `49152`、sector `1572864`；原始数据和分析已记录在
  `07_softmax/README.md`。V0 基线的正确性、Benchmark、Profile 与 Notes
  已验收，不代表 Day 9 最终验收。
- 已创建可编译的 V1 Shared Memory Softmax 框架：每行一个 block、每 block
  256 线程，预置 max/sum shared 数组和与 V0 共用的正确性、Benchmark、
  单次 Profile 入口；block 内两次规约和写回仍由学习者实现。当前空 kernel
  编译通过，未使用变量警告和正确性 `FAIL`（退出码 1）符合模板预期。
- 学习者已实现 V1 的两次 shared-memory tree 规约和归一化。17 个 shape
  正确性全部 PASS、退出码 0；memcheck 为 0 errors、0 bytes leaked。
- V0/V1 交替三轮 Benchmark：`(128,4096)` 的平均 latency 为
  `1.253107/0.025122 ms`，V1 约快 `49.882×`；`(128,1024)` 约快
  `66.570×`。详细原始数值已记录在 `07_softmax/README.md`。
- V1 的 Nsight Compute Profile 为 `39.49 µs`、global-load request
  `49152`、sector `1572864`，与 V0 相同，均为 `32 sector/request`。
  初版性能收益不能归因于访存合并；随后已针对列映射复测。
- 学习者已将 V1 三遍循环统一改为 `i=tid; i<hidden; i+=256`。
  修正版 17 个正确性用例全部 PASS、退出码 0；memcheck 为
  0 errors、0 bytes leaked，退出码 0。
- 修正版 V1 三轮 CUDA Event Benchmark 中 `(128,4096)` 平均 latency
  `0.005178 ms`，相对初版的 `0.025122 ms` 约快 `4.852×`；
  Nsight Compute global-load request 仍为 `49152`，sector 从
  `1572864` 降至 `196608`，即 `32→4 sector/request`。读取侧
  coalescing 改善得到 Profile 支持。V1 性能闭环验收完成。
- 学习者补充尝试在求和阶段把 `expf(x-max)` 暂存到 `output`，归一化
  阶段读回，以额外一次 output 写入换取源码层面少一次 `expf`。
  17 个正确性用例全部 PASS；本次未提供该变体的 Sanitizer 输出。
- 暂存变体三轮 Event Benchmark 中 `(128,4096)` 平均 `0.009092 ms`，
  相对已验收 V1 基线的 `0.005178 ms` 延迟增加约 `75.6%`；其余三个
  测试形状也均回退。global-load request/sector 仍为 `49152/196608`；
  这次优化没有带来性能收益，详细对照已记入 `07_softmax/README.md`。
- 学习者表示已对暂存版补跑正确性及 Compute Sanitizer，但目前只提供
  命令，尚无退出码和错误/泄漏摘要；这项安全复验暂不标记为通过。
- 已核对当前 `v1.cu`：归一化阶段已恢复为直接从 input 读取并重新
  计算 `expf`，不再把中间值暂存到 output；可继续作为 V2 对照基线。
- 已创建 V2 Warp Shuffle Softmax 可编译模板与中文任务说明。保留
  V1 的连续列扫描、CPU Reference、17 个 shape、CUDA Event 和
  单 kernel Profile 入口；warp max/sum 及跨 warp 规约留给学习者实现。
  CMake 默认构建 V1/V2，V0 目标归档为注释。
- 本地 CMake 配置及 `softmax_v1`/`softmax_v2` 构建通过；恢复后的
  V1 重新运行 17 个 shape 全部 PASS、退出码 0。空 V2 模板的
  17 个 shape 均按预期 FAIL、退出码 1；未使用的 partial/shared
  变量产生编译 warning，完成 TODO 后应消失。
- 已将已验收的 V0/V1 学习者源码单独提交，并把本地 `main` 推送到
  `github.com/trump253/cuda-bootcamp` 的 `main`。`origin` 拉取使用
  HTTPS、推送使用已认证的 SSH；用户已明确要求后续每次本地提交后
  自动推送。仍只提交当前任务范围内的文件，不顺手纳入其他工作区改动。
- 学习者已实现 V2 warp max/sum 与跨 warp 规约；17 个 shape 全部
  PASS、程序退出码 0，Compute Sanitizer 为 0 errors、0 bytes leaked，
  退出码 0。full mask 参与的无效 lane 分别用 `-INFINITY` 与 `0`，
  max/sum 各有两处 block barrier 保证 partial 与最终标量可见。
- V1/V2 同轮交替三次 CUDA Event Benchmark：`(128,1024)` 平均
  `0.004939/0.004148 ms`，V2 约快 `1.191×`；`(128,4096)` 平均
  `0.007364/0.006722 ms`，V2 约快 `1.095×`。四个测试形状均显示
  V2 更快，本轮与旧会话的 V1 绝对时长不混算。
- 学习者完成 V1/V2 最小 Nsight Compute 对照：shared load 指令
  `34816→2304`（少约 93.38%），store 指令 `18432→2304`
  （少 87.50%），barrier stall 比例 `8.02%→6.97%`；Profile
  duration `11.26→10.40 µs`。这些证据支持 V2 优化，但不能
  单独分解全部加速原因。原始数据和计数推导已记入 `07_softmax/README.md`。
- 已对照当前编译后的 shared LDS/STS 指令，推得 V1 load/store 为
  `128×2×8×(8×2+1)=34816` / `128×2×8×(1+8)=18432`；
  V2 两项均为 `128×2×9=2304`。这里是 warp 级指令数，
  不是逐线程访存元素数，也不是实际 shared 请求字节数。V2 的
  Reference、Correctness、Benchmark、Profile、Notes 闭环验收完成。
- 已澄清 V2 最终行标量读取的广播范围：同一 warp 的 lane 可从
  一条 shared load 获得同一值，但 8 个 warp 仍各执行一条读指令；
  V1 公式中的外层 `8` 是每 block 的 warp 数，括号中的 `8`
  是 tree 规约轮数，带谓词的指令计数不同于真正参与的线程数。

## V2 验收时的下一任务（历史记录）

- 下一步按学习计划准备 V3 合理向量化 Softmax 任务；仍由学习者
  完成核心 kernel，保持 V2 的正确性、Benchmark 与 Profile 对照。

## V2 验收时尚待确认的问题（历史记录）

- Day 1 的 GPU/CPU 索引变量类型尚未完全统一为 `std::size_t`。
- Day 8 的 `N=1024` 短 kernel 多轮结果有明显波动；若以后需要与 Release
  构建对照，须在新配置下重新采集，不能混用两种构建的数值。
- V0 的三个 Profile 指标能证明 load 访问高度分散，但不能单独量化 DRAM
  实际传输量，或分离不合并访存、低并行度与指数运算的耗时贡献。
- 本轮交替测量中，V0 每个尺寸的第一轮明显高于后两轮，不能直接用
  全部三轮均值作修正版 V1 的精确速度比基线。
- V1 列映射同时改变 global load 和 store 的线程地址分布；现有指标
  能证明 load sector/request 降低，不能单独量化 load 对全部加速的贡献。
- 暂存 `expf` 变体的 Compute Sanitizer 摘要与退出码尚未收到；
  output store 与缓存路径也未 Profile。目前足以判断没有观测到
  性能收益，但不足以确定唯一的回退原因。
- V2 源码仍保留模板期的 `TODO` 与“占位值”注释，虽然实际代码已经
  实现并通过测试；由学习者后续清理注释即可，不影响本轮性能结论。

## 历次关键知识

- kernel 在设备端执行，由主机端代码启动。
- 一维全局线程编号将 grid 中的 block 和 thread 映射到向量元素。
- 当 `N` 不能被 block 大小整除时，block 数量需要向上取整。
- 最后一个 block 中多出的 thread 必须进行边界检查。
- kernel 启动是异步的，启动错误和执行期错误都需要显式检查。
- GPU 输出与 CPU 参考结果必须存放在不同缓冲区中，否则参考计算会覆盖待验证
  结果，使正确性检查失效。
- 数值结果正确不能排除越界访问；需要结合 Compute Sanitizer 才能闭合正确性验证。
- 二维 grid 的 x 方向覆盖列、y 方向覆盖行；`1000 × 513` 配合
  `block=(16,16)` 时，`grid=(33,63)`。
- Matrix Add 每个元素读取 8 字节、写入 4 字节，只做一次加法，算术强度很低，
  因此属于典型的 memory-bound kernel。
- 异步 kernel launch 周围的 CPU wall-clock 若没有同步，通常只测到提交时间；
  CUDA Event 更适合测量 GPU stream 中的 kernel 执行时间。
- kernel-only latency 与包含分配、传输的端到端 latency 是两个不同指标。
- 有效带宽是算法字节数除以时间，不等于真实 DRAM 流量；cache 命中会影响两者关系。
- 正确性通过不代表优化成立；必须检查线程到地址的映射并用 Benchmark 验证。
- Shared memory 的 bank 由地址映射决定；二维数组第二维长度会改变跨行访问的
  地址步长，因此添加一列 padding 可以改变转置读取时的 bank 分布。
- 当前 `16 × 16` block 中一个 warp 跨两个 thread row，因此 padding=1 会将
  严重的转置读取冲突降为少量 2-way 冲突，但不像经典 `32 × 8` block 那样
  让每个 warp 固定在同一行并完全消除冲突。
- 本阶段中 bank conflict 专指 shared-memory bank 的访问冲突；global-memory
  访问主要使用 coalescing、sector 和 memory transaction 等概念描述。
- Nsight Compute 的 bank-conflict 计数与“几路冲突”不是同一个表达：本次
  8-way conflict 显示为每 warp 7 个额外 conflict，恰好对应理想访问之外的
  7 轮额外服务。
- 对短时 kernel，小尺寸测量更容易受固定开销与频率波动影响；优化结论应优先
  参考多轮稳定的大尺寸数据。
- GEMM 中同一线程跨 K 轮读取相邻 A 元素体现时间局部性；warp 合并访存要看同一
  条 load 指令下各 lane 的地址。单线程 B 跨 N 与固定 K 时 B 的 warp 合并读取
  可以同时成立。
- Nsight Compute 的 L1/TEX global-load request/sector 不代表实际 DRAM 流量；
  低 DRAM throughput 与较高 SM throughput 也不足以单独判定 GEMM 的唯一瓶颈。
- Naive GEMM 的 `grid=(64,64)` 有 4096 个 block、每 block 8 个 warp，共
  32768 个 warp；`32768×1024×2=67,108,864` 与实测 global-load request 数吻合。
  这里 4096 是 block 数，不是 warp 数，request 数也不等于逐线程 load 数。
- Tiled GEMM 的 A 同一行值由不同输出列线程复用，B 同一列值由不同行线程复用；
  第一次 barrier 保护 tile 写后读，第二次保护读后覆写。尾部补 0 保证无效 K 项
  不参与点积，并清除 shared memory 中上轮的残值。
- V2 Tiled GEMM 将 K 维划为 64 个 tile 阶段，实测 L1/TEX global-load request
  比 V1 少 16 倍；sector 总数约少 8.06 倍，因每个请求平均涉及的 sector 更多。
  DRAM throughput 百分比升高不能直接推出 DRAM 总传输量增加。
- 单线程负责计算、warp 发出合并访存请求、block 协作复用 shared tile。V2 将
  每个 warp 对每个操作数的 K 维 global-load 轮次从 1024 降到 64，但每个输出
  仍需沿 K 做 1024 次乘加，减少的不是总计算量。
- cuBLAS 对照要在同一 GPU 上交替测量相同 FP32 形状，并只比较正常运行的 CUDA
  Event 时间；本次 `1024³` cuBLAS 相对手写 V2 约快 6.772×。仅凭速度差不能推断
  库内部采用了哪一种具体优化。
- cuBLAS 的 6.772× 只适用于本次同条件 `1024³` 测量，不能推广到所有 GEMM。
  行主序内存可按列主序的转置矩阵解释，利用 `Cᵀ=Bᵀ×Aᵀ` 交换调用操作数，
  无需另行执行转置 kernel。
- Day 8 的 Vector Add 只有一次加法，不是“FP32 accumulation”练习；标量 FP16
  与 half2 都采用 FP16 加法，避免把不同计算精度混入向量化性能对照。
- GFLOP/s 与 GB/s 都由同一次 latency 计算；低算术强度的简单算子通常优先看
  有效带宽，数据可复用的 GEMM 通常优先看吞吐算力，但指标选择不等于瓶颈证明。
- 对同一个 kernel，算法 GFLOP/s ÷ 算法 GB/s = 算法 FLOP/字节；二者不是
  两次独立的硬件计数，也不能直接替代 DRAM 流量或已执行浮点指令的 Profile。
- half2 API 的可用性取决于编译目标架构，而不只取决于机器上插了什么 GPU；
  当前 CUDA 11.8 默认 `sm_52`，实际 GPU 是 `sm_75`，需要显式匹配。
- CMake 3.16 可用 `target_compile_options` 给当前 CUDA 目标统一传 `-arch=sm_75`；
  VS Code 的 C/C++ 扩展可从 `compile_commands.json` 读取实际编译选项。
- 当架构配置应覆盖整个学习工程时，可用目录级 `add_compile_options`，并以
  `COMPILE_LANGUAGE:CUDA` 限制只影响 CUDA 源文件；不会改变外部直接调用 nvcc
  的命令。
- 同尺寸延迟比是旧版本平均 latency / 新版本平均 latency；half2 当前
  `N=2^24` 相对 FP16 标量约 `1.086×`。相同 `N` 下，FP32 算法字节数是
  FP16 的两倍，因此 GB/s 略高不表示 latency 更短。
- half2 一线程处理两个 half，block 仍为 256 线程，但大尺寸 grid 从
  `ceil(N/256)` 变为 `ceil(N/512)`；请求数约减半、单请求 sector 数约
  翻倍，总 sector 数不变。
- CMake 的 Debug/Release 是配置名称，不能直接推断设备 kernel 是否失去优化。
  本次 nvcc `-g` 只生成主机调试信息；真正要特别警惕的是 `-G`，它在未启用
  `--dopt` 时会关闭设备代码优化。性能比较应记录并保持编译参数一致。
- Day 8 是从 FP32 基础 kernel 过渡到 FP16 数据表示、half2 双元素处理与
  精度/性能权衡的桥梁；未来 Softmax、归一化和 CUDALM 需要复用这种
  dtype、尾部边界、正确性与 Profile 思路，但本节 Vector Add 没有多项累加，
  不应误称已经实现了 FP32 accumulator。
- 已用当前 half2 的 `N=2^24`、`grid=32768`、`block=256` 和每 warp 两次
  global load，推得 `32768×8×2=524288` 个 L1/TEX load request，
  与学习者 Nsight Compute 实测一致；这是 Profile 合理性检查，不等于
  DRAM 访问数，也不能替代 sector/request 分析。
- 已推导 Day 8 的 L1/TEX global-load sector：half2 为
  `32768×8×2×4=2097152`，标量 FP16 为 `65536×8×2×2=2097152`。
  其中 4/2 是每个 warp 对单份输入连续读取 128/64 字节所覆盖的 32 字节
  sector 数；这个推导依赖本例的对齐、连续访问及完整 warp，不能把
  sector 数直接当作 DRAM 传输次数。
- Day 9 的数值稳定 Softmax 先按行求最大值，再计算 `exp(x-max)` 并归一化；
  减去最大值不改变结果，但可避免大正数输入导致指数溢出。V0 采用一线程一行
  作为正确性基线，暂不做行内并行规约。
- 当前 V0 的 `max(float,float)` 是 CUDA 设备端数学重载，行为对应 `fmaxf`；
  `exp(float)` 也在设备端执行，可显式写为 `expf`。CPU Reference 中的
  `std::max` 与 `std::exp(double)` 则在主机端执行；函数名不等于性能证据。
- V0 不必为了“使用 CUDA 函数”替换现有 `max`、`exp`；如需明确 FP32
  类型意图，可写 `fmaxf`、`expf`，但这不是已验证的加速，仍需先完成测试。
- V0 在 `(128,4096)` 使用 1 个 block、4 个 warp，每 warp 的相邻线程负责
  相邻行；同一列的 load 地址相隔 16384 字节，所以一次请求覆盖 32 个
  不同 sector。3 遍输入读取对应 `4×4096×3=49152` 个 load request，
  实测 `1572864/49152=32 sector/request`；这是 L1/TEX 计数，不是 DRAM
  读量。单线程跨列连续读取是时间上的局部性，不是同一 warp 指令的合并访存。
- 一个线程负责连续 16 列，不代表 warp 读取连续：V1 初版同一轮相邻 lane
  分别访问第 0、16、32…列，`hidden=4096` 时地址间距 64 字节，
  与 V0 一样触及 32 个 sector。大幅加速可由行内/跨 block 并行解释，
  不能用“初版 V1 合并访存已改善”解释。
- V1 修正版让同一轮的相邻 lane 负责相邻列，每线程下一轮跨 256 列。
  `(128,4096)` 的 L1/TEX global-load `sector/request` 从 32 降到 4，
  与连续读取 32 个 FP32 元素覆盖 4 个 32 字节 sector 相符；
  三轮 Event 平均延迟从 `0.025122 ms` 降到 `0.005178 ms`。
- Profile 的 `10.94 µs` 是 ncu 条件下的时长，正常运行的 Event
  `0.005178 ms` 才用于同口径 Benchmark；不同工具的绝对时长不直接混比。
- V1 的局部值写入 shared 后需要 barrier，保证所有线程完成初始化；
  tree 每轮末的 barrier 保证本轮写入对下一轮读取可见。max 与 sum
  两次规约均遵循此规则。barrier 不是错误检查，也不跨 block 同步。
- 把指数结果写到 output 再读回没有减少总 global-load 次数：
  直接重算版每元素 3 load + 1 store + 2 次源码 `expf`，暂存版
  3 load + 2 store + 1 次源码 `expf`。实测后者更慢，说明少算一次
  昂贵函数不自动等于整个 kernel 更快。
- Softmax V2 仍保持每行一个 256 线程 block：8 个 warp 分别规约
  局部值，warp 0 再规约 8 个 partial；跨 warp 传递仍需要 shared
  memory 和 block barrier。本轮 Event 已证明同条件下 V2 更快。
- ncu 的 `smsp__inst_executed_op_shared_{ld,st}` 计数单位是 warp
  级 SASS 指令；当前 V1 的带谓词 tree 指令可由 warp 发出，
  即使部分 lane 的谓词为假。V2 每次规约每 block 只有 9 条
  shared load 与 9 条 store 指令，不能把这些数解释为字节数。
- Shared memory 广播只发生在同一 warp 的请求内：多个 lane 读
  同一地址不会形成 bank conflict，但不会把不同 warp 的 8 条
  load 指令合成一条。V1 的 load 公式括号 `8×2+1` 分别对应
  8 轮、每轮 2 条 shared load、最终 1 条标量 load。
- 已清理 V0/V1/V2 源码中完成项的过时 TODO 和占位说明，补充行映射、
  规约单位元、shared-memory barrier 与 V2 两级规约的中文注释；
  V2 实现与已验收的正确性、Benchmark、Profile 记录一同归档。
  清理后 V1/V2 重新编译并各通过 17 组正确性；V2 Compute Sanitizer
  复验为 0 errors、0 bytes leaked，退出码为 0。
- 已创建 V3 FP32 `float4` 练习框架，复用 V2 已验收的两级规约、
  主机端 Reference/正确性/Benchmark/Profile；只把对齐行的三遍向量化
  访问与尾部处理留给学习者，未对齐行提供 V2 标量路径。共用测试新增
  `(1,2)`、`(5,6)`，覆盖余数为 2 的尾部与行首对齐变化。
- 默认 CMake 保留 V2/V3 作同条件对照，V1 target 归档为注释。
- V2/V3 目标均编译通过；扩展测试后 V2 的 19 组正确性全部 PASS、
  退出码为 0，Compute Sanitizer 为 0 errors、0 bytes leaked。
  V3 未填写 TODO 时 19 组均按预期 FAIL、退出码为 1，因此目前
  只是任务模板，尚未达到正确性或性能验收。
- 学习者已完成 V3 FP32 `float4` 主体和尾部实现；19 组正确性全部
  PASS、退出码 0，Compute Sanitizer 为 0 errors、0 bytes leaked。
  本地重新构建、正确性与 memcheck 复验一致。
- V2/V3 交替三轮 Event 对照：`(128,4096)` 平均分别为
  `0.006681/0.006811 ms`，V3 约慢 1.95%；`(16,512)` V3
  约快 5.24%，不同尺寸没有一致收益。一次 ncu 对照中 V3 的
  global-load request 为 V2 的四分之一（49152→12288），
  sector 均为 196608；只证明请求粒度改变，不证明 DRAM 字节数减少。
- 学习者已正确推导 `hidden=33` 时 row 0～5 的路径分别为
  向量化/标量/标量/标量/向量化/标量，并解释向量化减少 request、
  逻辑输入量不变。结合实测的 4→16 sector/request 和总 sector
  不变，V3 FP32 的概念解释与 Notes 已完成；负收益如实保留。
- 已清理 `07_softmax/v3.cu` 六处完成项的过时 TODO，仅修改中文注释，
  保留向量组映射、尾部标量处理和规约说明，不改变算法。
- 按学习计划将 FP16/half2 任务限定为 FP16 输入/输出存储、FP32
  max/exp/sum/归一化，并提供同 dtype 的标量 FP16 基线与 half2
  练习版。主机测试复用 Day 9 的输入生成和 CPU Reference，先量化
  输入再生成参考值，覆盖原 19 个形状及 CUDA Event/Profile 路径。
- `softmax_fp16_scalar` 和 `softmax_fp16_half2` 均编译通过；标量基线
  19 组正确性全部 PASS、退出码 0，Compute Sanitizer 为 0 errors、
  0 bytes leaked、退出码 0。half2 模板的六处 TODO 尚未实现，
  19 组按预期 FAIL、退出码 1，未进行性能验收。
- 学习者提交 half2 初版后有 15/19 组 PASS，`(1,1)`、`(16,33)`、
  `(128,257)`、`(129,33)` 失败。Review 定位到 aligned 分支的
  指数和尾部循环误将 `expf(...)` 加到 `thread_max`，而实际规约
  `thread_sum`；这是漏算尾部导致的分母错误，不是 FP16 阈值过严。
- 学习者已修正 half2 的 sum 尾部。FP16 标量与 half2 各 19 组
  正确性全部 PASS；half2 的 Compute Sanitizer 报告 0 errors、
  0 bytes leaked。本地构建、正确性与 memcheck 复验一致。
- FP16 标量/half2 交替三轮 Event 中，`(128,4096)` 平均 latency
  分别为 0.006536/0.006190 ms，half2 约快 5.30%；但 `(1,128)`
  和 `(128,1024)` 的 half2 分别约慢 5.13% 和 5.70%，没有一致收益。
- 一次 `(128,4096)` ncu 对照：标量/half2 global-load request
  为 49152/24576，sector 均为 98304，Profile 时间为 8.51/8.45 µs。
  request 减半只说明访存分组变化，不说明读取字节数或 DRAM 流量减半。
  V3 FP16 的正确性、Benchmark、Profile、Notes 已闭环。
- 已建立 Day 10 中文任务说明和学习者填写的时间线记录模板；复用
  Vector Add、Softmax V2 与 Reduction V3，不新增学习者需要实现的 kernel。
- 已给 Reduction V3 添加仅精简主机测试流程的 `--timeline` 入口：
  `N=1048579` 依次产生 `4097 → 17 → 1` 个 partial sum。三个
  Day 10 目标构建、直接运行均通过；`--timeline` 输出 `passes=3`、
  `abs_error=0`、`PASS`。
- 本机检测到 Nsight Systems 2022.4.2.50；冒烟采集成功生成
  `.nsys-rep`，`cudaapitrace`、`gputrace`、`kernexectrace` 报表
  命令均可运行。冒烟检查只验证工具链，不替代学习者的正式采集与分析。
- 2026-10-08：已核对 NVIDIA 官方 Nsight Streamer 文档；它可在
  服务器容器内运行 Nsight Systems GUI，通过浏览器查看已有 `.nsys-rep`。
  当前容器检测到 Ubuntu 20.04、x86_64、RTX 2080 Ti，PATH 中未找到
  Docker/Podman；尚未安装或启动 Streamer，也未验证浏览器连接。
- 学习者确认只能使用当前容器。检查发现 `/.dockerenv` 存在、控制组
  路径含 `kubepods`；无 Docker socket，也未配置 `DOCKER_HOST`。
  初次排查时 `nsys-ui` 已在 PATH 中，Xvfb、VNC server、noVNC 启动命令未找到。
  当时的两份报告各约 150 KiB，首先讨论了下载到本地 GUI 的方案。
- 2026-10-08：已清理复习问答正文和列表项内的硬换行；格式整理阶段
  从 870 行合并到 478 行，Markdown 解析核对确认正文、代码块、
  标题、列表与链接保持一致；随后补充了本次 GUI 缺库排查问答。
- 已修复 GUI 的 `libOpenGL.so.0`、`libEGL.so.1`、报告插件所需
  NSS/NSPR 库及 JPEG 插件依赖。主程序、CrashReporter 与 xcb
  平台插件的动态库检查通过，QuadDPlugin 已能加载。
- 已在当前容器安装并启动 Xvfb、Openbox、x11vnc、noVNC；Mesa
  llvmpipe 的 OpenGL 3.1 检查通过。HTTP 页面返回 200，WebSocket
  升级返回 101，RFB 帧缓冲与鼠标输入验证通过。
- 已实际打开 `build/day10_vector_add.nsys-rep`，确认 GUI 中显示
  时间线与 CUDA API/GPU 行；新增 `08_nsys/GUI.md` 记录 SSH / VS Code
  端口转发、环境变量、缺库修复及重启步骤。此项验证不替代学习者分析。

## 当前问题

- Day 10 的三份正式报告与四个时间线问题仍待学习者完成；
  Day 11 与最终独立 Softmax 验收尚未进行。
- 容器侧 GUI 和浏览器中转已经验证；学习者本地浏览器的 SSH
  端口转发与交互显示仍待确认。当前入口为远程端口 `6080`。

## 今日关键知识

- `float4` 每组覆盖四个相邻 FP32 元素；必须检查行首实际地址的
  16 字节对齐，尾部不足四个元素不能直接用完整 `float4` 读取。
- 某行是否对齐还取决于 `row`：基址对齐时条件为
  `(row * hidden) % 4 == 0`。`hidden % 4 == 0` 足以保证所有行
  对齐，但不是某一行对齐的必要条件。
- 对照 V2/V3 时固定 dtype、block、规约算法和测试口径，先验证正确性，
  再用 Event 与 Profile 判断访存变化和性能收益。
- 本次 V3 的 L1/TEX global-load request 降为四分之一，但每请求
  sector 从 4 增到 16、总 sector 不变；请求数下降不等于整体加速。
- 逻辑读取字节数不变本身不能保证所有实现的 sector 数都相同；
  本次 sector 相同还与两版对齐且覆盖相同输入区域的访问模式相符。
- FP16 Softmax 的主机参考先量化输入，再以量化后的真实数值求 double
  Softmax；核内以 FP32 求 max、`expf`、sum，最后写回才舍入为 FP16。
- `half2` 的任务是成对访存，不是用 FP16 `__hadd2` 求指数和；
  输入/输出行首需满足 4 字节对齐，奇数列尾部用标量处理。
- 尾部元素必须分别参与 max、指数和与写回；sum 阶段误写到
  `thread_max` 会使分母漏掉尾部。`hidden=1` 时分母变为 0；
  其他奇数列可能因尾部贡献很小而偶然通过阈值，不能因此认为正确。
- FP32 max/exp/sum 既维持标量/half2 实验的算术口径，也避免长行
  指数和反复 FP16 舍入；max 仅选择已量化输入中的最大值，并非
  数学上必须用 FP32。是否改成 FP16 算术属于另一项独立实验。
- 基址对齐时，`hidden=33` 的 FP16 行首每跨一行偏移 66 字节；
  偶数行满足 half2 的 4 字节对齐，16 对后剩 1 个标量尾部；
  奇数行整行走标量路径。代码实际检查输入、输出两个行首。
- half2 把 global-load request 减半，但每请求覆盖 sector 从 2
  增至 4，总 sector 不变；Event 的快慢依形状而异，不能把
  request 减少直接等同于整体速度收益。
- Nsight Systems 区分 CPU 上的 CUDA API 调用与 GPU 上的 kernel/
  数据传输：两侧时间段不是同一个概念。第一次 CUDA 调用可能含
  上下文初始化或模块加载，不宜当作稳态 launch overhead。
- Reduction V3 的多轮规约在同一 stream 中顺序执行；Day 10 要
  通过实际时间线确认三轮顺序与 gap，并把“观察到 gap”和
  “解释 gap 的具体原因”区分开。
- Nsight Streamer 将服务器上的 GUI 画面传到浏览器，报告通过目录
  挂载留在服务器上。当前官方镜像支持 VP9 软件编码；RTX 2080 Ti
  可选择此模式，AV1 GPU 编码加速才要求 Ada 或更新架构。
  查看报告的 GUI 版本应与采集端相同或更新。
- 容器内找不到 Docker 不代表宿主机没有 Docker；单独安装 Docker
  客户端也不等于能创建容器。Docker bind mount 的源路径属于 daemon
  所在宿主机，不能直接假定当前容器的报告路径就是宿主机路径。
- `nsys-ui` 的 OpenGL 版本 `0` 可能来自显示探测失败，不能据此
  判断显卡能力。需分别检查动态库、显示入口和渲染能力；运行时
  加载的报告插件还可能有主程序 `ldd` 未覆盖的依赖。
- Xvfb 提供容器内虚拟屏幕；VNC/noVNC 与 SSH 转发让本地浏览器
  能访问它。当前软件渲染经 `glxinfo` 和 GUI 日志验证为 OpenGL 3.1。

## 下一任务

- 按 `08_nsys/GUI.md` 转发远程端口 `6080`，在本地浏览器连接
  已启动的 GUI，完成 Vector Add 报告的时间线观察。
- 学习者按照 `08_nsys/README.md` 亲自采集 Vector Add、Softmax V2、
  Reduction V3 三份报告，填写 `08_nsys/NOTES.md`，提交关键截图
  或 CLI 片段及四个问题的答案供 Review。达到 Day 10 验收后再进入
  Day 11 单 kernel Nsight Compute 对照。
