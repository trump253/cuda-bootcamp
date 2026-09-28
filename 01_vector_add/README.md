# Day 1：向量加法

## 状态

已完成并通过验收：

- 所有规定规模的 CPU/GPU 最大绝对误差均为 0。
- Compute Sanitizer 未发现非法内存访问。
- CUDA 内存泄漏为 0 字节。

## 已覆盖知识

- Host 与 Device
- Grid、Block 与 Thread
- 一维全局线程编号
- block 数量向上取整
- 尾部边界保护
- CUDA API、kernel 启动与执行期错误检查

## 保留的 Review 建议

`main.cu` 中 GPU 和 CPU 的两个索引仍使用 `int`，而元素数量使用
`std::size_t`。请自行统一类型，并确保 block 偏移的乘法本身也在
`std::size_t` 类型下执行。
