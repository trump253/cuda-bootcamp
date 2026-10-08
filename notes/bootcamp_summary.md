# Day 12：CUDA Bootcamp 综合复盘

状态：待学习者回答。Day 11 基础验收已通过，最终独立 Softmax 验收尚未开始。本任务对应学习计划第 12 节的复盘，不新增 kernel，也不添加新的 CUDA 主题。

## 今天的任务与验收

先用自己的话回答下面 10 个已经遇到的问题，每题约 2–4 句话；涉及实现或性能判断时，结合至少一个已有练习或实验说明。可查看自己的源码与历史数据，再用 [复习问答](../CUDA_复习与面试问答.md) 查漏补缺，不直接复制整段答案。

不确定的地方可以明确写出，导师只针对实际缺口补讲。尤其注意区分性能观测与原因推断，不能把最大 stall、低 DRAM 吞吐或高 occupancy 单独当作瓶颈证明。

完成后提交这个文档或直接在对话中回复。验收看概念、例子和证据边界，不要求背长指标名；涉及数值时保留 ms、µs、GB/s、GFLOP/s、% 等对应单位。导师 Review 后再准备从空源文件开始的最终独立 Softmax 任务，不复制旧 kernel。

## 1. CUDA execution model 是什么？

回答：待填写。

## 2. global、shared、register 分别解决什么问题？

回答：待填写。

## 3. coalescing 为什么重要？

回答：待填写。

## 4. warp reduction 是怎么工作的？

回答：待填写。

## 5. 什么是 memory-bound？

回答：待填写。

## 6. 什么是 compute-bound？

回答：待填写。

## 7. 如何 benchmark CUDA kernel？

回答：待填写。

## 8. Nsight Systems 和 Nsight Compute 分别解决什么问题？

回答：待填写。

## 9. 为什么不能只看 occupancy？

回答：待填写。

## 10. 什么时候应该开始 CUDALM？

回答：待填写。说明进入条件，以及当前还缺哪项验收证据；不要仅因 Day 11 通过就宣称已完成 Bootcamp。

## Review 记录

- 学习者提交：待完成。
- 概念与例子：待 Review。
- 需要补充的实际缺口：待 Review。
- 下一任务：复盘通过后，准备最终独立 Softmax 验收；RMSNorm/fusion 仍为可选补缺，不自动追加为必做任务。
