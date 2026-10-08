# Day 10 时间线观察记录

2026-10-08：学习者已提交三份报告的 GUI 截图和 Vector Add、Softmax 初步解释，随后补交 Reduction GPU 事件截图与依赖/开销分析。以下数值由仓库中的原始报告复核；gap 与开销口径已在 Review 中更正，Day 10 时间线练习验收完成。

## 环境与报告

- GPU / 工具版本：NVIDIA GeForce RTX 2080 Ti；Nsight Systems 2022.4.2。
- 构建配置与三个程序的退出码：待填写。
- Vector Add 报告：`build/day10_vector_add.nsys-rep`。
- Softmax V2 报告：`build/day10_softmax_v2.nsys-rep`。
- Reduction V3 报告：`build/day10_reduction_v3.nsys-rep`。

## Vector Add：传输与同步

- 选取最后一个测试尺寸 `N=1048579`，两个 H2D 分别上传 A、B。
- 以下时间戳以报告开始为原点，单位统一为 µs；不是墙上时钟时间。

| 事件 | CPU API 开始（µs） | CPU API 持续（µs） | GPU 开始（µs） | GPU 持续（µs） | GPU 开始 − API 开始（µs） |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2D A，CorrID 171 | 818749.235 | 438.559 | 818846.580 | 428.925 | 97.345 |
| H2D B，CorrID 172 | 819188.234 | 451.797 | 819355.729 | 373.918 | 167.495 |
| Vector Add launch，CorrID 173 | 819641.387 | 15.940 | 819737.391 | 20.255 | 96.004 |
| D2H C，CorrID 176 | 819760.273 | 859.821 | 819770.095 | 642.075 | 9.822 |

学习者所读的 97.345/167.495 µs 是调用开始到 GPU 传输开始的间隔；传输本身的持续时间是 428.925/373.918 µs。最后的 D2H 位于 `cudaDeviceSynchronize` 返回之后。

`cudaDeviceSynchronize` 从 819658.072 µs 开始，持续 101.907 µs，在 819759.979 µs 返回。进入同步时第二次 H2D 仍在执行，因此同步时间包含先前排队的传输与 kernel 等待，不等于 kernel 的 20.255 µs。按时间戳可拆成：kernel 开始前 79.319 µs、kernel 执行 20.255 µs、kernel 结束到 API 返回 2.333 µs；这只是时间区间划分，不能据此独立识别每段的所有内部开销。

报告将输入标记为 Pageable；普通 `std::vector` 存储的 H2D `cudaMemcpy` 可以在 GPU 传输完成前返回。本次 API 返回/GPU 传输结束分别为 A 的 819187.794/819275.505 µs、B 的 819640.031/819729.647 µs，与该行为一致。

## Softmax V2：一次 launch 与执行

- CPU launch 开始：773245.892 µs；持续：166.040 µs；返回：773411.932 µs。
- GPU kernel 开始：773410.031 µs；持续：8.640 µs。
- 截图 `Latency` 是 164.139 µs，而非初步回答的 64.139 µs；它等于 `773410.031 − 773245.892`，即调用开始到执行开始。
- GPU 在 API 返回前 1.901 µs 已开始执行，两侧时间段部分重叠。异步启动不要求 GPU 必须等 CPU API 返回后才能开始工作。
- 本模式仅启动一次目标 kernel；166.040 µs 不能直接作为稳态 launch 开销，初始化/模块加载及采集扰动的贡献尚未分离。

## Reduction V3：多 kernel 序列

- 报告确认三轮，grid 为 `4097 → 17 → 1`，block 均为 256 线程。本次截图未附程序最终 PASS 输出；kernel 数值正确性沿用先前已完成的 V3 验收，Day 10 核对时间线概念。
- 三次 launch 均在 Stream 7。后轮读取前轮的 partial sum，数据依赖要求顺序；代码使用同一 stream，提供该顺序保证。CPU 可连续提交，无须在每两轮之间调用 `cudaDeviceSynchronize`。

| 轮次 | grid.x | CPU launch 持续（µs） | GPU Start（µs） | GPU Duration（µs） | GPU End（µs） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1，CorrID 111 | 4097 | 115.090 | 811537.041 | 17.920 | 811554.961 |
| 2，CorrID 113 | 17 | 8.824 | 811555.698 | 2.879 | 811558.577 |
| 3，CorrID 115 | 1 | 6.211 | 811559.345 | 2.816 | 811562.161 |

按 `gap = 下一轮 GPU Start − 本轮 GPU End`，两处 gap 分别为 `811555.698 − 811554.961 = 0.737 µs`、`811559.345 − 811558.577 = 0.768 µs`。学习者初步读出的 4/3 µs 已更正；GUI 中以秒显示的 Start 被舍入，不宜用这种显示精度直接计算亚微秒间隔。gap 存在的事实已确认，当前记录没有将其具体成因独立分解。

后两轮的 CPU launch 持续时间分别约为 GPU Duration 的 3.065/2.206 倍，主机提交成本相对显著。但 GPU 的约 2.8 µs 是设备执行区间，不能说其中“基本都是 CPU 提交”。17 个 block 可以分布在多个 SM 并行执行，每 block 的规约流程与 1 个 block 版本相近；因此总 kernel 持续时间不必随 block 数线性增长。小规模设备执行还包含访存、规约、同步、调度与采集影响，当前数据不能单独求出“纯计算”占比。

CPU 在第一轮 GPU 执行时提交后两轮，第三轮提交还与第二轮 GPU 执行部分重叠。CPU launch Duration 不能与 GPU Duration 直接相加来代表整个序列耗时；第一次 115.090 µs 的 launch 也不代表稳态提交成本。

## 四个验收问题的结论

1. kernel launch 是否有 gap：Softmax 的 start-to-start latency 已确认；Reduction 相邻 GPU kernel 的 gap 为 0.737/0.768 µs。
2. H2D/D2H 在哪里：Vector Add 已定位并复核，见上表。
3. 短 kernel 的主机提交/等待开销是否显著：后两轮 launch 为 8.824/6.211 µs，GPU 执行为 2.879/2.816 µs，提交成本相对显著；主机与设备有重叠，不能据此将整个序列归为提交时间。
4. 多 kernel pipeline 的执行顺序：同一 stream 保证前轮完成后后轮执行，满足 partial sum 的依赖；数据依赖本身不自动建立跨 stream 顺序。

## 本次仍不确定的问题

- 亚微秒 gap 的内部原因及设备执行区间中纯计算的占比，当前证据未分解。
- 运行配置、程序退出码和 Reduction 最终 PASS 输出尚未在本次提交中给出。

字段含义参考 [NVIDIA Nsight Systems latency 与 overhead 说明](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)；pageable H2D 的返回语义参考 [CUDA 11.8 API 同步行为](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/api-sync-behavior.html)。stream 顺序与 block 调度参考 [CUDA 11.8 编程指南](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#streams)。
