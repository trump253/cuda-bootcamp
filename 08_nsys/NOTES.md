# Day 10 时间线观察记录

2026-10-08：学习者已提交三份报告的 GUI 截图和 Vector Add、Softmax
初步解释。以下数值由仓库中的原始报告复核；Reduction 的完整解释
与短 kernel 开销判断仍由学习者继续完成。

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

学习者所读的 97.345/167.495 µs 是调用开始到 GPU 传输开始的
间隔；传输本身的持续时间是 428.925/373.918 µs。最后的 D2H
位于 `cudaDeviceSynchronize` 返回之后。

`cudaDeviceSynchronize` 从 819658.072 µs 开始，持续 101.907 µs，
在 819759.979 µs 返回。进入同步时第二次 H2D 仍在执行，因此
同步时间包含先前排队的传输与 kernel 等待，不等于 kernel 的
20.255 µs。按时间戳可拆成：kernel 开始前 79.319 µs、kernel
执行 20.255 µs、kernel 结束到 API 返回 2.333 µs；这只是时间区间
划分，不能据此独立识别每段的所有内部开销。

报告将输入标记为 Pageable；普通 `std::vector` 存储的 H2D
`cudaMemcpy` 可以在 GPU 传输完成前返回。本次 API 返回/GPU
传输结束分别为 A 的 819187.794/819275.505 µs、B 的
819640.031/819729.647 µs，与该行为一致。

## Softmax V2：一次 launch 与执行

- CPU launch 开始：773245.892 µs；持续：166.040 µs；
  返回：773411.932 µs。
- GPU kernel 开始：773410.031 µs；持续：8.640 µs。
- 截图 `Latency` 是 164.139 µs，而非初步回答的 64.139 µs；
  它等于 `773410.031 − 773245.892`，即调用开始到执行开始。
- GPU 在 API 返回前 1.901 µs 已开始执行，两侧时间段部分重叠。
  异步启动不要求 GPU 必须等 CPU API 返回后才能开始工作。
- 本模式仅启动一次目标 kernel；166.040 µs 不能直接作为
  稳态 launch 开销，初始化/模块加载及采集扰动的贡献尚未分离。

## Reduction V3：多 kernel 序列

- 程序输出的 `passes` 和最终结果：待填写。
- 三轮 kernel 的 GPU 执行顺序与各轮 grid：待填写。
- 相邻 kernel 间是否有 gap：待填写，标注单位。
- 从时间线可以确认的事实与仍属推断的原因：待填写。

学习者已经提交截图；下一步请选中 GPU 行的三个 kernel，读取
各自的开始时间和持续时间，再计算
`gap = 下一轮 GPU 开始 − 本轮 GPU 结束`。CUDA API 行上同名
事件的 Duration 是主机 launch 调用持续时间，不能代替 GPU Duration。

## 四个验收问题的结论

1. kernel launch 是否有 gap：Softmax 的 start-to-start latency 已确认；
   Reduction 的相邻 GPU kernel gap 与解释待学习者补全。
2. H2D/D2H 在哪里：Vector Add 已定位并复核，见上表。
3. 短 kernel 的主机提交/等待开销是否显著：待填写，附数值与单位。
4. 多 kernel pipeline 的执行顺序：待填写。

## 本次仍不确定的问题

- Reduction 三轮 GPU 开始/持续时间及 gap，最后两轮主机调用与
  GPU 执行时间的比较，待学习者填写。
- 运行配置、程序退出码和 Reduction 最终 PASS 输出尚未在本次提交中给出。

字段含义参考 [NVIDIA Nsight Systems latency 与 overhead 说明](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)；
pageable H2D 的返回语义参考 [CUDA 11.8 API 同步行为](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/api-sync-behavior.html)。
