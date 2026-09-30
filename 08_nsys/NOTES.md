# Day 10 时间线观察记录（由学习者填写）

## 环境与报告

- GPU / `nsys --version`：待填写。
- 构建配置与三个程序的退出码：待填写。
- Vector Add 报告：`build/day10_vector_add.nsys-rep`。
- Softmax V2 报告：`build/day10_softmax_v2.nsys-rep`。
- Reduction V3 报告：`build/day10_reduction_v3.nsys-rep`。

## Vector Add：传输与同步

- 选取的测试尺寸：待填写。
- CPU 上的 H2D API 调用、GPU 上的数据传输、kernel、D2H、同步各在何处：待填写。
- 对照截图或 CLI 输出片段的位置：待填写。

## Softmax V2：一次 launch 与执行

- CPU launch 调用时间：待填写，标注单位。
- GPU kernel 执行时间：待填写，标注单位。
- launch-to-start 是否有可辨认的间隔：待填写；若无法判断，说明原因。

## Reduction V3：多 kernel 序列

- 程序输出的 `passes` 和最终结果：待填写。
- 三轮 kernel 的 GPU 执行顺序与各轮 grid：待填写。
- 相邻 kernel 间是否有 gap：待填写，标注单位。
- 从时间线可以确认的事实与仍属推断的原因：待填写。

## 四个验收问题的结论

1. kernel launch 是否有 gap：待填写。
2. H2D/D2H 在哪里：待填写。
3. 短 kernel 的主机提交/等待开销是否显著：待填写，附数值与单位。
4. 多 kernel pipeline 的执行顺序：待填写。

## 本次仍不确定的问题

- 待填写；没有则写“无”。
