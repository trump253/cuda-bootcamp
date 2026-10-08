# Day 10：Nsight Systems 时间线练习

本节已完成采集与 Review，验收通过；原始时间戳、口径更正和
结论见 [NOTES.md](NOTES.md)。以下命令保留供复现与复习。

本节只按学习计划练习系统级时间线：CPU 发起 CUDA 调用、H2D/D2H、
GPU kernel 与相邻 kernel 之间的空隙。复用已验收的 kernel，今天
**不需要写新 kernel，也不以优化吞吐量为目标**。

## 最少理论

- CPU 上的 kernel launch 通常只是把工作提交给 GPU；CUDA API 调用
  的持续时间不等于 GPU kernel 的执行时间。
- 同步式 `cudaMemcpy` 的 H2D/D2H 和 `cudaDeviceSynchronize` 可能让
  主机等待。时间线可帮助定位这些事件，但不能只凭一段空隙就认定
  唯一原因是 launch overhead。
- Nsight Systems 用来看跨 CPU/GPU 的顺序与间隔；此前的 CUDA Event
  主要计量指定 GPU 工作，Nsight Compute 主要分析单个 kernel。
  Profile 会扰动执行，不把其中的绝对耗时直接当作普通运行的基线。
- 首次 CUDA 调用可能包含上下文初始化，首次 kernel 启动还可能
  包含模块加载；不要拿它们直接代表稳态 launch overhead。
- 查看字段时区分三种时间：CPU API 的 Duration 是调用持续时间，
  GPU 事件的 Duration 是设备工作持续时间，Latency 是关联的
  GPU 工作开始时间减去 CPU API 开始时间。CPU API 与 GPU 工作
  可以部分重叠，不能把这些时间直接相加。
- 本练习的 Vector Add 使用 pageable 主机输入；H2D `cudaMemcpy`
  返回时 GPU 传输可能仍在执行，后面的设备同步会等待这些工作。
  语义见 [CUDA 11.8 API 同步行为](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/api-sync-behavior.html)。

## 实验对象与接口

| 程序 | 命令 | 今天观察什么 |
| --- | --- | --- |
| Day 1 Vector Add | `./build/vector_add` | H2D、kernel、D2H 与主机同步 |
| Day 9 Softmax V2 | `./build/softmax_v2 --profile` | 单个 Softmax kernel 的 launch 与执行；此模式没有 D2H |
| Day 5–6 Reduction V3 | `./build/reduction_v3 --timeline` | 同一输入上的三轮依赖规约，以及 H2D/D2H |

`reduction_v3 --timeline` 只运行一次 `N=1048579` 的正确性用例，
第一、二、三轮的输出数分别是 `4097 → 17 → 1`；程序应输出
`passes=3`、`PASS`。这个入口只精简主机测试流程，没有修改规约 kernel。

## 第一步：构建与直接运行

```bash
cmake -S . -B build
cmake --build build --target vector_add softmax_v2 reduction_v3 -j 8
CUDA_VISIBLE_DEVICES=0 ./build/vector_add
CUDA_VISIBLE_DEVICES=0 ./build/softmax_v2 --profile
CUDA_VISIBLE_DEVICES=0 ./build/reduction_v3 --timeline
```

先确认程序能正常退出；如没有 `nsys`，用 `command -v nsys` 和
`nsys --version` 检查本机环境。当前仓库所在环境检测到
Nsight Systems 2022.4.2.50；以下选项已按本机 `nsys profile --help`
与 `nsys stats --help-reports` 核对。

## 第二步：你自己采集三份时间线

报告写入已被 Git 忽略的 `build/` 目录。命令不使用强制覆盖选项；
再次运行时给 `--output` 加新后缀，以保留原始报告。

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none --output=build/day10_vector_add \
  ./build/vector_add

CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none --output=build/day10_softmax_v2 \
  ./build/softmax_v2 --profile

CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none --output=build/day10_reduction_v3 \
  ./build/reduction_v3 --timeline
```

可在 Nsight Systems 图形界面打开 `.nsys-rep`。当前 SSH 容器已有通过浏览器访问 `nsys-ui` 的方案，连接、缺库修复与重启步骤见 [GUI.md](GUI.md)。

也可用本机提供的 CLI 报表辅助查看事件与时间戳：

```bash
nsys stats --report cudaapitrace --report gputrace \
  --report kernexectrace --timeunit usec \
  build/day10_reduction_v3.nsys-rep
```

`cudaapitrace` 看 CPU 发起的 CUDA API，`gputrace` 看 GPU 上的
kernel/内存操作，`kernexectrace` 用于关联 launch 与执行。其他
两份报告可把末尾文件名改为相应的 `day10_*.nsys-rep`。如果你的
`nsys` 版本不支持其中某个报表，先运行 `nsys stats --help-reports`
查本机可用名称，不要猜数字。

## 你要自己回答并提交给我 Review

在 [NOTES.md](NOTES.md) 填入观察值和结论，并附三份报告的关键截图
或上述 CLI 的相关片段。至少回答：

1. Vector Add 中 CPU 的 `cudaMemcpy` 与 GPU 上的 H2D/D2H 各在哪里？
   `cudaDeviceSynchronize` 位于哪一段？
2. Softmax 的 CPU launch 调用和 GPU kernel 在时间线上是否完全重合？
   有无可辨认的 launch-to-start 间隔？
3. Reduction 的三轮 kernel 按什么顺序执行？相邻轮之间是否有 gap？
   能否从记录中区分“看到 gap”与“已确认 gap 的成因”？
4. 对短 kernel，CPU 侧提交/等待时间相对于 GPU 执行时间是否显著？
   用你看到的数值和单位说明，不要只凭印象。

验收标准：三个程序能运行并生成报告；能在时间线上指出 H2D、
kernel、D2H、CPU launch、同步和多轮执行顺序；`NOTES.md` 含
明确的时间单位与证据，不把 API 耗时直接说成 kernel 耗时。
