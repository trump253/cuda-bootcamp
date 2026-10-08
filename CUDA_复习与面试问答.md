# CUDA Bootcamp 复习与面试问答

本文只记录本仓库已经实际学习、实现、测量或讨论过的问题。后续每完成一个阶段，
继续补充对应知识点；尚未验证的性能判断必须明确标为“待验证”，不提前扩展课程范围。

## 1. CUDA 执行模型与索引

### Host 和 Device 分别是什么？

Host 是 CPU 及其主机代码，负责准备数据、分配内存和启动 kernel。Device 是 GPU，
kernel 实际在 Device 上执行；kernel launch 默认对 Host 异步。

### 一维线程如何映射到数据？

```cpp
global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x` 指定 block，`threadIdx.x` 指定 block 内线程，两者共同得到 grid 内的
一维全局线程编号。

### 为什么 block 数量要向上取整？

```cpp
grid_size = (n + block_size - 1) / block_size;
```

当 `n` 不能整除 `block_size` 时，需要额外一个 block 覆盖尾部元素。多出的线程
必须执行边界检查。例如 `N=1000, block=256` 时需要 4 个 block，最后一个 block
只有 232 个线程有效，24 个线程越界。

### 二维矩阵如何映射？

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

对于 row-major 矩阵，线性地址为 `row * cols + col`。二维输入的行、列都需要独立
做边界检查。

## 2. 正确性与错误检查

### 为什么 GPU 输出和 CPU Reference 必须使用不同缓冲区？

如果 CPU Reference 覆盖了待验证的 GPU 结果，比较过程可能永远“通过”，失去验证
意义。两者必须独立保存，然后计算误差。

### 数值结果正确是否足以证明 kernel 安全？

不够。越界访问可能暂时没有改变可见结果。当前流程同时要求：

- CPU Reference 对比；
- NaN/Inf 检查；
- CUDA API 和 kernel 错误检查；
- Compute Sanitizer memcheck；
- 泄漏检查。

### `cudaGetLastError()` 和同步分别检查什么？

`cudaGetLastError()` 用于检查 launch 配置等启动错误。kernel 异步执行期间发生的
错误，需要通过 `cudaDeviceSynchronize()`、Event 同步或后续同步 API 才能暴露。

### Compute Sanitizer 下的性能数据能否用于 Benchmark？

不能。Sanitizer 会插桩并显著改变执行时间，只用于检查非法访问和泄漏。真实性能
必须使用正常运行时的 CUDA Event Benchmark。

## 3. Benchmark 与带宽

### CPU wall-clock 是否表示 kernel 在 CPU 上运行？

不是。它只是由 CPU 计时器观察经过时间。异步 launch 后若不进行同步，通常只测到
提交时间；加入 `cudaDeviceSynchronize()` 后，还会包含 Runtime/Driver 调用、队列
等待和主机阻塞开销。

### 为什么 kernel latency 不包含 `cudaMalloc` 和 H2D/D2H？

kernel-only latency 与端到端 latency 是不同指标。前者只衡量 GPU kernel，后者
可以包含分配和传输。两种指标都有意义，但必须分开命名和测量。

### 为什么使用 CUDA Event？

CUDA Event 被记录在 GPU stream 的执行序列中，可以测量 start 与 stop 之间的 GPU
工作。当前流程在计时前 warm-up，在计时区外完成分配和复制，并对多次 kernel 的
总时间取平均。

### 有效带宽是什么？

```text
effective_bandwidth = 算法定义的字节数 / kernel 时间
```

它适合比较同一算法的不同版本，但不等于真实 DRAM throughput。重复使用相同缓冲区
时可能命中 cache。所有输出必须显式携带单位，例如 `0.259670 ms`、`516.88 GB/s`。

### 为什么 GEMM 主要报告 GFLOP/s，而 Vector Add 等主要报告 GB/s？

两个数都由同一个 kernel 时间推得：`GFLOP/s = 算法 FLOP 数 / 时间`，
`GB/s = 算法字节数 / 时间`。选择主指标是为了表达当前算子的主要工作。

FP32 Vector Add 每元素读 A、读 B、写 C，共约 12 字节，只做 1 次加法，
算术强度约 `1/12 FLOP/字节`；矩阵转置甚至没有浮点运算。因此这些简单
算子通常优先报告有效带宽。GEMM 则做约 `2MNK` 次浮点运算；A/B 数据经
tile、cache 等复用后，同一批数据能服务许多乘加，因此通常优先报告
GFLOP/s。报告 GEMM 的 GB/s 也可以，但必须说明采用哪一种字节统计口径；
算法输入输出字节数不等于实际发生的所有内存访问或 DRAM 流量。

选用 GFLOP/s 不等于已经证明 GEMM 一定是 compute-bound；选用 GB/s 也不
等于已经证明其他 kernel 只受 DRAM 限制。要判断瓶颈，需结合算术强度、
缓存/内存流量和 Profile。跨不同算子比较时，始终先看同口径的 latency。

### 每个 kernel 都有计算吞吐和带宽，之前只报告一个是否不准确？

对有浮点运算且有数据读写的 kernel，两种速率可以同时报告；只选主指标是
展示习惯，不表示另一个资源未被使用。用相同的“算法 FLOP 数”和“算法字节数”
定义时，`算法 GFLOP/s ÷ 算法 GB/s = 算术强度（FLOP/字节）`。例如 FP32
Vector Add 为 `1/12 FLOP/字节`；有效带宽若为 500 GB/s，对应算法吞吐约
41.7 GFLOP/s。这两个数字来自同一次 latency，并非两次独立测量。

但须区分“算法指标”和“硬件实测”：上述 GB/s 是按读 A、B、写 C 的必要
字节数计算的有效带宽，不等于 L1/L2/DRAM 任一级的实际传输速率；算法
GFLOP/s 也不等于执行了多少条浮点硬件指令。矩阵转置没有浮点运算，其
GFLOP/s 为 0、没有分析价值。要定位真正瓶颈，还需结合 Profile 中各层
内存流量、指令和占用情况，不能只看这两个派生速率。

### 当前 GPU 的理论显存带宽是多少？

RTX 2080 Ti 的有效数据率为 14 Gbit/s、总线宽度为 352 bit：

```text
14 × 352 ÷ 8 = 616 GB/s
```

Matrix Add 在 `4096 × 4096` 上测得约 555.82 GB/s，对应算法有效带宽约为理论值
的 90.2%。当前程序只使用一张 GPU，不能把两张卡的理论带宽相加比较。

### 为什么需要暖机和多轮重复？

短 kernel 容易受启动固定开销、GPU P-state 和升频过程影响。Reduction V0 的首次
测试中，小尺寸前两轮比暖机后慢约 32%；丢弃一次整程序暖机后，大尺寸三轮极差降到
0.04% 以下。因此性能结论应使用相同条件下的多轮稳定结果。

## 4. Global Memory、Coalescing 与 Sector

### 什么是合并访存？

一个 warp 中相邻活跃线程访问相邻地址时，硬件可以用较少的 memory transaction
服务请求。跨大步长访问会触及更多 sector，增加传输量和事务数量。

### Sector 是什么？

在当前 global-memory 分析中，sector 可以理解为 cache line 内更小的传输/请求
单位，常以 32 bytes 分析。一个线程只读取 4-byte `float` 时，仍可能占用一个
sector；相邻线程访问连续 `float` 可以共同利用较少 sector。

### Naive Transpose 为什么慢？

Naive 映射中，`threadIdx.x` 连续变化使 input 读取连续，但 output 地址以输入
`rows` 为步长变化，写入不合并。Shared-memory tile 先合并读取，再交换 block/
thread 的写回职责，使 global output 也连续。

## 5. Shared Memory、Bank 与 Padding

### `__syncthreads()` 的作用范围是什么？

它是 block 级 barrier，并保证 barrier 前的 shared-memory 写入对 block 内线程
可见。它不能同步不同 block。若 barrier 位于非收敛控制流中，行为未定义，可能
挂起或得到错误结果，不能简单认为一定死锁。

### Shared-memory bank 如何计算？

对于当前 FP32 场景，可用以下近似分析：

```text
linear_index = row * stride + col
bank         = linear_index % 32
```

`stride` 是二维数组相邻两行起点之间的元素数，即声明的第二维长度；padding 列不
存放逻辑数据，但会改变 stride 和 bank 映射。

### `[16][16]` 转置读取为什么产生 8-way conflict？

`16 × 16` block 的一个 warp 跨两个 thread row。读取
`tile[threadIdx.x][threadIdx.y]` 时，stride=16 使 warp 集中访问 bank
0、1、16、17，每个 bank 对应 8 个不同地址，因此形成 8-way conflict。

### `[16][17]` 是否完全消除 conflict？

没有。当前 block 形状下，padding=1 后 shared load 和 store 各剩一个 2-way
conflict。经典 `32 × 8` block 中一个 warp 固定在同一行，配合 stride=33 才能让
转置读取覆盖 32 个不同 bank。

### Padding 的优化证据是什么？

`4096 × 4096` Transpose：

- latency：0.280313 ms → 0.259670 ms，降低约 7.36%；
- 有效带宽：478.81 GB/s → 516.88 GB/s；
- Nsight Compute shared load conflict：每 warp 7 → 1；
- load+store 总 conflict：每 warp 7 → 2，降低约 71.43%。

“8-way conflict”描述服务需要约 8 轮，而该计数器记录理想第一轮之外的额外轮次，
所以显示为每 warp 7 个 conflict。

## 6. Reduction

### 为什么多个线程不能直接更新同一个 `output[0]`？

普通加法更新是读取—修改—写回过程。多个线程并发更新同一地址会产生数据竞争，
可能读取相同旧值并互相覆盖。需要 block 内 reduction、分级 partial sum 或原子操作
等安全汇总方式。

### Reduction V0 为什么带宽极低？

V0 只启动一个 block，七个 warp 全部退出，warp 0 也只有 lane 0 串行循环。只有
一个 SM 获得工作，没有足够的活跃 lane、warp 或 block 来形成有效合并访问、隐藏
内存延迟或利用多个 SM。`N=2^20` 仅约 0.212 GB/s。

### 为什么 Reduction 需要多轮 kernel？

`__syncthreads()` 只能同步一个 block。第一轮每个 block 只能产生一个 partial sum；
host 再启动相同 kernel 归约 partial sums。相邻 kernel 位于同一 stream，后一轮会在
前一轮完成后执行，从而形成 grid 范围的阶段边界。

### 为什么使用两个 partial buffer ping-pong？

同一轮中不同 block 没有全局 barrier。若输入和输出重叠，先完成的 block 可能覆盖
其他 block 尚未读取的数据。使用 A/B 两个缓冲区交替可避免读写覆盖。

### 为什么尾部线程写 0，而不是提前返回？

0 是加法单位元，不改变结果；同时所有线程必须继续到达后续 block barrier。部分
线程提前返回而其他线程执行 `__syncthreads()` 不符合 barrier 收敛要求。

### Tree Reduction 为什么每轮都要同步？

当前轮线程写入 shared memory 后，下一轮线程会读取这些新值。barrier 保证本轮写入
完成并对下一轮可见，避免读取旧值。

### V1 的交错寻址在源码层面有什么问题？

V1 使用从 1 开始逐轮翻倍的 stride。第一轮偶数 lane 满足条件，第二轮每四个 lane
只有一个满足，之后继续稀疏；同时每轮包含取模判断。源码层面存在交错的线程条件，
但不能只看 CUDA `if` 就断言硬件一定生成 divergent branch：编译器可能使用
predication。V1 在 `N=2^20` 上为 0.048428 ms、86.61 GB/s。

### V2 准备如何减少 divergence？

V2 从 `blockDim.x/2` 开始逐轮减半 stride，并让低编号的连续线程参与：先是
thread 0～127，再是 0～63、0～31。前几轮可以让整个 warp 全部活跃或全部不活跃。
V1/V2 同条件 Benchmark 已验证该优化：`N=2^20` 从 0.048428 ms 降到
0.033143 ms，加速约 1.461×；`N=2^24` 从 0.639414 ms 降到 0.432722 ms，
加速约 1.478×。

`block=256` 时，stride=128、64、32 分别对应 4、2、1 个完整活跃 warp，其余
warp 对分支条件统一为 false，不属于 warp 内分歧。从 stride=16 开始，warp 0
只有 lane 0～15 活跃，出现一个分歧 warp；stride=8 时只有 lane 0～7 活跃。

### Nsight Compute 是否证明 V2 的 divergent branch 更少？

没有。在 RTX 2080 Ti 上，V1/V2 的 divergent branch target 都是 0，uniform
branch 比例都是 100%，说明短条件被编译器主要转换为 predication 等形式。

Profile 证明的实际差异是：V2 的 warp instructions 从 129,302,528 降到
53,805,056，减少 58.39%；predicated-on thread instructions 减少 65.05%。两版
shared load/store bank conflict 都是 0。因此当前证据支持“V2 显著减少执行指令且
没有引入 bank conflict”，不能简单表述成“硬件 divergent branch 数下降”。

### `__shfl_down_sync` 如何完成当前的 warp reduction？

同一 warp 的 lane 可以直接读取更高 lane 的寄存器值，delta 按 16、8、4、2、1
递减并逐轮相加，最终 lane 0 得到 warp sum。Shuffle 不能跨 warp，因此一个
256-thread block 仍需让 8 个 warp leader 把 partial sum 写入 shared memory，
经过一次 `__syncthreads()` 后，再由 warp 0 做第二级归约。

### Full mask 是否会让无效或已退出的 lane 参与执行？

不会。mask 声明参与这次 shuffle 的 lane 集合，不会激活不存在或已经退出的线程。
当前使用 full mask 安全，是因为每个 warp 的 32 个真实 lane 都会执行 shuffle；
越界线程不提前退出，只把输入贡献设为加法单位元 0。第二级归约同理：warp 0 的
32 个 lane 全部执行，只有前 8 个 lane 读取有效 warp sum，其余 24 个 lane 提供 0。

### `__shfl_down_sync` 能否替代 `__syncthreads()`？

不能。`__shfl_down_sync` 在同一 warp 内完成 mask 所指定 lane 的参与/收敛与寄存器
交换，但它不是通用 shared/global memory fence，也不能同步不同 warp。8 个 warp
leader 写完 `warp_sums` 后，仍需一次 `__syncthreads()`，warp 0 才能安全读取。

### V3 更快是否只是因为 shuffle 比 shared memory 快？

不能只归因于单条指令的速度。V3 把大部分 warp 内数据交换留在寄存器中，使每个
block 使用的 shared memory 从 V2 的 256 个 float 降到 8 个 float，并把 block
barrier 从 V2 的初始化后一次加 8 轮归约同步，降到 warp 间交接所需的一次；同时
也减少了 shared load/store 和控制工作。大尺寸 Benchmark 已证明整体 V3 更快：
`N=2^20` 加速约 1.688×，`N=2^24` 加速约 1.750×。

Nsight Compute 进一步证明 V3 相对 V2：总 warp instructions 减少 66.63%，shared
load/store instructions 分别减少 99.22%/88.89%，barrier stall 比例从 22.41%
降至 6.39%，累计等待 barrier 的 warp 数减少 87.25%。这些是整体优化成立的直接
证据，但不能据此精确拆分每项变化各自贡献的时间。

### 为什么 barrier 从 9 次降到 1 次，stall 指标没有恰好下降 9 倍？

源码中的静态 barrier 次数不等于动态等待量。每次 barrier 的等待时间还取决于各
warp 到达时间、调度和可被其他工作隐藏的程度；百分比指标又以 active-warp 周期为
分母。本次累计等待数从 246,446,359 降到 31,432,461，约下降 7.84 倍，方向明确，
但不要求等于 9 倍。Shuffle 自身有执行成本，不过它不是 barrier stall 指标未按
9:1 缩放的直接原因。

### 为什么小规模 Reduction 的 V3 加速比较小？

`N=1024` 的首轮只有 4 个 block，不能填满 GPU 的全部 SM；同时 kernel launch、
多轮调度等固定成本在约 5 微秒的总时间里占比较高，掩盖了 block 内优化。大规模
输入能提供足够多的 block，每个 block 节省的 shared-memory、barrier 和指令开销
在大量 block 上累积，因此 `N=2^24` 的 V3/V2 加速扩大到约 1.750×。block 增多
不会让单个 block 本身自动变快。

### Reduction 的复杂度和性能属性是什么？

Reduction 的总工作量仍为 `O(N)`，因为约有 `N-1` 次加法；`O(log N)` 描述的是
并行 tree reduction 的关键路径深度。每个输入至少读取一次，而每个元素只对应约
一次加法，算术强度很低，因此大规模 Sum Reduction 通常偏 memory-bound。

当前 V3 有效带宽约 271.36 GB/s，低于 RTX 2080 Ti 的 616 GB/s 理论带宽，且减少
同步与指令也带来了明显收益。因此不能把当前实现描述为“只受 DRAM 带宽限制”；
同步、归约依赖和指令开销仍然存在。

## 7. Naive GEMM

### 一个线程如何计算一个 GEMM 输出元素？

对于 row-major 的 `A=(M,K)`、`B=(K,N)` 和 `C=(M,N)`，二维线程映射到
`C[row,col]`，再沿 K 维计算：

```text
C[row * N + col] += A[row * K + inner] * B[inner * N + col]
```

输出边界由 `row < M && col < N` 保护。

### `M=31, N=33, block=(16,16)` 的 grid 和右下角利用率是多少？

grid.x 覆盖 N 列，等于 `ceil(33/16)=3`；grid.y 覆盖 M 行，等于
`ceil(31/16)=2`，所以 grid 是 `(3,2)`。右下角 block 剩余 15 个有效行、1 个
有效列，只有 15 个有效线程，241 个线程空闲。

### Naive GEMM 中 A/B 的 warp 级访存模式是什么？

当前 block 为 `16×16`，一个 warp 跨两个 thread row。对固定 inner：每个
16-thread 半 warp 的 A load 访问同一个 `A[row,inner]`，形成广播；B load 则随
连续 col 读取 `B[inner,col]` 的连续地址，可以合并。两个半 warp 对应不同输出行，
但会读取相同的 B 行片段。

不能根据“单个线程在 K 循环中 A 连续、B 跨 N”判断 coalescing；coalescing 关注
一个 warp 在同一条内存指令上的地址分布。A 标量会被多个输出列线程复用，B 标量
会被多个输出行线程复用，这为 shared-memory tiling 提供了机会。

### 同一线程下一轮读取相邻 A 元素，是否叫合并访存？

不叫。合并访存观察的是同一个 warp 的活跃线程执行同一条 load 指令时的地址分布；
同一线程在 K 循环中先后读取 `A[row,inner]`、`A[row,inner+1]`，属于跨时间的相邻
访问，更适合用时间局部性描述。这些元素可能由先前的内存事务带入 cache，从而让
后续访问命中 cache，但跨轮次的多个标量 load 不会因此自动变成一次 warp 合并访存。

对同一线程，B 地址随 inner 增加而跨 N；同时对固定 inner，同一 warp 的 B 读取
仍可沿 col 合并。这两个描述的观察方向不同，可以同时成立。

### 为什么 GEMM 通常按 `2MNK` 计算 FLOPs？

每个输出元素进行 K 次乘法和累加，性能报告约定一次 multiply-add 计 2 FLOPs，
因此每个输出约 `2K` FLOPs，全部输出约 `2MNK` FLOPs。严格数学计数可以写成
`K` 次乘法加 `K-1` 次加法，但 Benchmark 通常使用前一种行业惯例。

### 为什么参考值接近 0 时最大相对误差可能很大？

相对误差的分母是参考值绝对值。参考值接近 0 时，即使绝对误差只有约 `1e-8`，
比值也可能非常大。因此当前 GEMM 使用混合条件：

```text
abs_error <= absolute_tolerance + relative_tolerance × |expected|
```

接近 0 时由绝对容差保护，数值较大时相对容差才更有意义。

### Naive GEMM 的首次 Nsight Compute 结果说明什么？

在 `M=N=K=1024`、`block=(16,16)`、`grid=(64,64)` 的单次 kernel Profile 中，
GPU duration 为 2.35 ms，global-load request 为 67,108,864，global-load sector 为
134,216,439，平均约 2 个 sector/request；SM throughput 为 62.50%，DRAM throughput
为 0.90%。request/sector 是 L1/TEX 路径上的请求与 sector 计数，不能直接当作
实际 DRAM 读流量或 cache 命中率。

这里 `grid=(64,64)` 表示 `64×64=4096` 个 block；每个 block 有 `16×16=256`
个线程，也就是 8 个 warp，因此一共启动 `4096×8=32768` 个 warp。每个 warp
遍历 1024 次 K 循环，每轮分别加载 A、B。测得的 global-load request 数恰好为
`32768×1024×2=67,108,864`，与当前 kernel 的两条加载路径吻合。这是 warp
层面的请求计数；不能误读为只启动 4096 个 warp，也不能把它等同于单个线程的
load 次数。每个请求还可能涉及多个 sector，本次平均约 2 个。

当前 `--profile` 模式把 A/B 初始化为 0；V2 必须使用相同模式做对照。较低的 DRAM
throughput 与数据复用、cache 服务等现象相容，但不能只凭这组聚合指标断言某个
单一瓶颈，SM throughput 也不等同于浮点计算单元利用率。

### Tiled GEMM 中 A/B tile 如何复用？

当前 `16×16` block 的一个线程计算一个 `C[row,col]`。在固定的 K 阶段和 tile
内部索引 `i` 下，`tile_a[threadIdx.y][i]` 被同一输出行的 16 个不同列线程
使用；`tile_b[i][threadIdx.x]` 被同一输出列的 16 个不同行线程使用。因此每个
阶段协作加载 A/B tile 后，可以在 block 内复用这些值。

### 不完整 tile 为什么补 0？两次 barrier 分别做什么？

越界的 A/B 元素写 0，保证 K 尾部无效项对点积没有贡献，并覆盖 shared memory
中上一阶段留下的值。不能让一部分线程在 `__syncthreads()` 前提前 return，否则
block barrier 位于不一致的控制流中，行为未定义。

第一次 barrier 保证所有线程写完当前 A/B tile，其他线程才开始读取；第二次
barrier 保证所有线程读完当前 tile，下一阶段才能覆写 shared memory。

### 反复使用 `threadIdx.x/y` 与先保存为 `tx/ty` 有性能差异吗？

两种写法在当前 kernel 中都正确。`threadIdx` 是 CUDA 提供的线程索引，不是每次
使用都要访问 global memory。把它们保存成局部变量更方便阅读，也能让数组索引
保持一致；现代编译器通常会消除重复读取。不能仅凭源码写法断言性能提升，若确实
需要比较，以编译结果与同条件 Benchmark/Profile 为准。

### V2 Tiled GEMM 的 Profile 是否证明了 A/B tile 复用？

同尺寸 `1024³` 对照中，V1 的 L1/TEX global-load request 为 67,108,864，V2
为 4,194,304，恰好减少 16 倍。V2 有 4096 个 block、每 block 8 个 warp、每
个 warp 经过 64 个 K tile 阶段并加载 A/B，因此
`4096×8×64×2=4,194,304`，与测量相符。

sector 数从 134,216,439 降到 16,656,832，约下降 8.06 倍；平均每 request
涉及的 sector 从约 2.00 增至约 3.97。两版的 warp 加载形状不同，所以 request
降幅与 sector 降幅不必相同。这些是 L1/TEX 路径上的计数，不能直接当作 DRAM
流量。正常 Benchmark 在 `1024³` 上从 2.348016 ms 降到 1.505280 ms，约加速
1.560×；Profile 说明加载请求大幅减少，是支持优化机制的直接证据，但无法单独
量化每项变化对 latency 的贡献。

### V2 的 DRAM throughput 比例更高，是否说明它读取了更多 DRAM 数据？

不能。该比例从 0.90% 增至 1.41%，但同时 kernel 时间从约 2.35 ms 降到
1.51 ms。吞吐比例描述单位时间内达到峰值的程度，不等于总传输字节数；当前
request/sector 指标也不测量 DRAM 字节。SM throughput 从 62.50% 增至 73.23%
同样是综合指标，不等于专门的 FP32 计算单元利用率。

### GEMM 分析访存时，该站在线程、warp 还是 block 的角度？

三个角度各有用途：单线程角度看点积的 K 次乘加和每轮需要哪些值；warp 角度看
同一条内存指令形成多少 global-load request/sector；block 角度看 A/B tile 如何
在不同输出线程之间复用、何时需要 barrier。不能只选一个角度解释全部性能。

当前 V1 的每个 warp 对每个操作数沿 K 发出 1024 轮 global load，V2 将 K 划为
64 个长度为 16 的阶段，每阶段每个 warp 对 A/B 各发出一次 global load；对应
L1/TEX request 由 67,108,864 降到 4,194,304，正好下降 16 倍。单线程的
global load 次数也随协作加载减少，但它仍需参与 1024 次乘加，并从 shared
memory 读取 tile 值。减少的是 global-load 请求，不是总计算量。这里 request
计数恰好符合 warp 加载轮次的公式；不能把这种一一对应推广成所有 kernel 的规则。

### cuBLAS SGEMM 与手写 Tiled GEMM 的一次对照结果是什么？

在相同 GPU、FP32、`1024³`、warm-up=5、iterations=20 的 CUDA Event 测量中，
手写 V2 三轮平均为 1.504731 ms、1427.15 GFLOP/s；cuBLAS 三轮平均为
0.222186 ms、9665.25 GFLOP/s，cuBLAS 约快 6.772 倍。五组形状包括非方阵，
都通过 CPU Reference；Compute Sanitizer 为 0 errors、0 bytes leaked。

这是库实现与手写基础 tiled kernel 的对照结果，不能仅凭 latency 推断 cuBLAS
内部采用了哪一种优化。Sanitizer 会插桩，不能用其运行时间比较性能。

### `6.772×` 表示 cuBLAS 在所有 GEMM 上都快这么多吗？

不表示。这是本次单卡 FP32 `1024³`、相同 CUDA Event 计时条件下的速度比：
`1.504731 ms ÷ 0.222186 ms ≈ 6.772`。也可以说 cuBLAS 在这次实验中的
GFLOP/s 约为手写 V2 的 6.772 倍，或耗时约减少 85.23%。换形状、硬件、
精度或实现后，比值都可能改变。

### 为什么行主序 GEMM 调用列主序 cuBLAS 时要交换 A/B？

矩阵乘积的转置满足 `(A×B)ᵀ=Bᵀ×Aᵀ`。同一段 row-major `A(M,K)` 内存，
可由 cuBLAS 按 column-major `Aᵀ(K,M)` 解释；B 和 C 同理。因此调用时交换
B/A 的位置来计算列主序的 `Cᵀ(N,M)`。结果写入的字节按 row-major `C(M,N)`
读取正好是所需的 C；这里没有执行额外的转置 kernel。

## 8. FP16 / half2

### 为什么 `__hadd` 能编译，`__hadd2` 却报告未定义？

当前 CMake 3.16 构建没有传 CUDA GPU 架构选项，CUDA 11.8 的 `nvcc` 默认
按 `sm_52` 编译。所装 `cuda_fp16.h` 仅在设备架构 `__CUDA_ARCH__ >= 530`
时声明 `__hadd2`；RTX 2080 Ti 的实际计算能力是 7.5。用 `-arch=sm_75`
单独编译当前 `half2.cu` 已成功，说明这次是编译目标架构与 API 能力不匹配，
不能据此判断 half2 kernel 的逻辑正确性。CMake 3.16 尚不支持
`CMAKE_CUDA_ARCHITECTURES`；需要对当前三个 Day 8 目标明确传相同的
`-arch=sm_75`。目前已改用 `add_compile_options` 在项目级只给 CUDA 编译
传这个选项，因此历史目标重新启用时也会继承；重新配置与构建成功，
后续正确性与 Sanitizer 均已通过。

### “同尺寸延迟比”怎么计算？为什么 FP32 的 GB/s 较高却不一定更快？

固定相同元素数 `N`、同一 GPU 和同样计时方式，用旧版平均 latency 除以新版
平均 latency；比值大于 1 表示新版更快。当前 Debug 构建 `N=2^24` 三轮平均：
FP32 0.362695 ms、FP16 标量 0.198085 ms、half2 0.182479 ms。因此 half2
相对 FP16 标量约快 `0.198085/0.182479≈1.086×`，latency 降低约 7.88%。

FP32 的有效带宽约 555.08 GB/s，half2 约 551.64 GB/s；但 FP32 每元素
的算法读写量为 12 字节，half2 为 6 字节。FP32 在同一个 `N` 上要处理
两倍的算法字节数，所以它的 latency 反而约为 half2 的 1.988 倍。
GB/s 是单位时间处理的字节数，不是“谁耗时最短”的直接排序。这批数据
来自同一 Debug 构建，比较条件一致；`N=1024` 的短 kernel 还出现明显波动。

### 为什么 half2 的 grid 约为 FP16 标量的一半？block 数不是相同吗？

当前 harness 固定 `block=256`，但 FP16 标量一线程处理 1 个 half，
half2 一线程处理 2 个 half。故 `grid.x`（block 数）分别为
`ceil(N/256)` 和 `ceil(N/512)`。例如 `N=2^24` 时分别是 65536 和
32768 个 block；每个 block 内仍都有 256 个线程。只有 `N` 较小或受
向上取整影响时，两者的 grid 才可能相同。总启动线程数在大尺寸下约
减半，活跃线程数也由约 `N` 降为约 `ceil(N/2)`；奇数 `N` 的最后一个
half 由一个线程单独处理。

### half2 的 global-memory request 减半、sector 不变说明什么？

在学习者采集的 `N=2^24` L1/TEX Profile 中，FP16 标量的 load/store
request 为 1,048,576/524,288，half2 为 524,288/262,144；load/store
sector 两版均为 2,097,152/1,048,576。平均每次请求涉及的 sector
由 2 增为 4：标量版同一 warp 的每个线程读取一个 2 字节 half，合计
64 字节；half2 版每个线程读取两个 half，合计 128 字节。因此请求数
减半但总请求 sector 数不变是合理的。sector 是对齐的 32 字节区域；
这些是 L1/TEX 计数，不是实际 DRAM 字节数。当前结果支持 half2 在
这次 Debug 对照中更快，但不能仅凭 request 变化认定它是唯一加速原因。

### Debug 构建的 CUDA Event 数据必须切到 Release 才能验收吗？

不能只看构建配置名称。当前 CMake 缓存为 Debug，实际 nvcc 命令是
`-g -arch=sm_75 -std=c++14`，没有 `-G`。CUDA 11.8 的 `-g` 生成主机代码
调试信息；`-G` 才生成设备调试信息，并且在未指定 `--dopt` 时关闭设备
代码优化。旧练习保留的构建参数也显示 `-g` 而无 `-G`，因此无法据此
认定以前的 Benchmark 无效。Day 8 三版在相同构建配置下交替测量，
当前的相对延迟结论可以验收。Release 可用于另做部署配置对照，但不是
本节的必选补测；不同配置的绝对数值不能直接混在同一组比较中。

### Day 8 的 FP16/half2 练习在后续 Softmax 和 CUDALM 中有什么用？

本节用最简单的 Vector Add 隔离三个实际问题：数值类型改变后，参考值和
误差判据也要重新定义；每线程处理两个相邻 half 时，grid、尾部边界、
访存请求和 latency 如何变化；性能优势必须在相同 `N`、相同精度与构建
条件下用 Benchmark/Profile 证明。当前标量 FP16 与 half2 的对照证明
了该实现的收益，并不意味着所有 FP16 算子都会同比例加速。

这些能力会迁移到后续 Softmax、归一化和 CUDALM 的 FP16 张量处理：
读写可用较低精度节省存储与搬运，连续元素可考虑成对处理；但涉及
多项求和等精度敏感步骤时，要单独选择计算/累加类型并验证误差。
当前 Vector Add 只有一次加法，没有实际练习 FP32 accumulation，也没有
证明 half2 可以直接替代所有 FP32 计算。

### half2 的 `L1/TEX global-load request = 524288` 能从代码推出来吗？

能，在当前特定 kernel 中可预测并用实测核对。Profile 固定 `N=2^24`，
half2 一线程处理两个 half，`block=256`、`grid=32768`；每个 block 有
`256/32=8` 个 warp。大尺寸恰好整除，不存在尾部空闲 warp。
源码中每个 warp 对 A、B 各执行一次 global load；在当前 Turing 的
L1/TEX LSU 路径中，每个 warp 的一次 global-load 指令产生一次 request。
因此 `32768 block × 8 warp/block × 2 load/warp = 524288 request`，
恰好等于实测。相同推理得到 store request 为 `32768×8×1=262144`。

这是一种有用的 Profile 合理性检查，但不是任何 kernel 都可照抄的公式：
编译器可能改变访存指令数，边界与分支可能让部分 warp 不执行 load，复杂
访问还涉及更多 sector、wavefront 或不同缓存层。request 也不是 DRAM
访问次数；要分析合并效果，还应看 sector/request。当前 half2 的 load
为 `2097152/524288=4 sector/request`，与每个 warp 连续读取
`32×4=128` 字节、每 sector 32 字节相符。

### L1/TEX global-load sector 数是怎么计算出来的？

sector 是对齐的 32 字节区域。当前 `N=2^24`、输入地址对齐、连续访问且没有
尾部空闲线程，因此可先算每个 warp 的一次 load 覆盖几个 sector，再乘以执行
load 的 warp 数和每个 warp 的 load 次数。这里计数的是 L1/TEX 收到的 sector
访问，不是 DRAM 实际传输量。

- half2：`grid=32768`，每 block 8 个 warp；每个 warp 对一份输入读
  `32×4=128` 字节，覆盖 `128/32=4` 个 sector。A、B 各有一次 load，故
  `32768×8×2×4=2097152` 个 global-load sector。
- 标量 FP16：`grid=65536`，每 block 8 个 warp；每个 warp 对一份输入读
  `32×2=64` 字节，覆盖 `64/32=2` 个 sector。A、B 各有一次 load，故
  `65536×8×2×2=2097152` 个 global-load sector。

也可以做字节数核对：两份输入共 `2×N×2=67108864` 字节，除以 32 正好
是 `2097152`。这个快捷算法只适用于当前对齐、连续、没有额外重复或浪费
sector 的访问；跨 sector 的未对齐地址、跨步访问、掩码线程或重复访存都可能
让实测 sector 数与“逻辑字节数/32”不同。应以每次 warp 访存触及的对齐
32 字节区域为准，而不能把 request、sector 与 DRAM 事务混为一谈。

## 9. Softmax

### kernel 中不写 `std::` 的 `max`、`exp` 是 CPU 标准库函数吗？

不是在 CPU 上执行。当前 V0 的 `row_max` 与 `input_row[i]` 均为 `float`；
CUDA 11.8 提供设备端的 `max(float, float)` 重载，其行为等价于 `fmaxf`。
`exp(float)` 也有设备端单精度重载，可明确写成 `expf(float)`。这些调用
在 kernel 的 GPU 设备代码中执行；编译器可能把它们内联或降低为多条设备
指令，不能仅凭源码名字断定具体指令序列或耗时。

作为对照，`softmax_harness.h` 的 CPU Reference 显式使用 `std::max`、
`std::exp`，且将输入转为 `double`，它们在主机端执行。若希望 GPU 源码的
精度意图更明显，可以显式使用 `fmaxf`、`expf`；不要把它们误认为会把数据
传回 CPU 的函数调用。仅凭函数名不能判断瓶颈，需结合实测。
对应函数定义可查 [CUDA 11.8 Math API](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-math-api/group__CUDA__MATH__SINGLE.html)。

### V0 有必要把 `max`、`exp` 改成 CUDA 函数吗？用哪个？

当前未限定命名空间的 `max(float,float)`、`exp(float)` 已在 GPU 端执行，
不必为了“改成 CUDA 函数”而替换。若想明确限定本版的 FP32 意图，可分别
使用 `fmaxf`、`expf`；这属于可读性和类型选择，不是已经证明的性能优化。
V0 应先固定一种写法，完成正确性、Benchmark 和 Profile，再考虑函数实现或
近似版本的对照；不能仅凭名字推断更快。当前 V0 基线已完成这三项实测。

### Softmax V0 为什么是 32 sector/request？三遍读取都来自 DRAM 吗？

`(rows,hidden)=(128,4096)` 时，V0 每线程负责一行，共 1 个 block、4 个
warp。对同一条读取指令和同一个列索引 `i`，warp 内相邻 lane 对应相邻行，
地址相隔 `hidden×sizeof(float)=4096×4=16384` 字节；32 个 lane 各自触及
一个不同的 32 字节 sector。因此一次 warp load request 涉及 32 个 sector，
而连续读取 32 个 FP32 元素的理想情况是 4 个 sector。单个线程下一轮读取
本行的相邻元素属于时间局部性，不会改变当前 warp 指令的合并情况。

源码有三遍输入读取（求 max、求指数和、写归一化结果），实测
`4 warp × 4096 列 × 3 遍=49152` 个 L1/TEX global-load request，乘以每
request 32 sector，得到 `1572864`，与学习者 Profile 完全一致。
这些计数表明三遍都发出了 global-load 请求，但不能证明每次都从 DRAM
取回；缓存命中也会在 L1/TEX 层被计数。V0 的 `grid=1`、每线程串行处理
4096 列及两遍 `expf` 也是可能的耗时因素。仅凭这三个指标不能断言 V0
单纯是 DRAM 带宽瓶颈，或量化每种因素的贡献。
sector 与 request 的定义见 [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

### 为什么 V1 初版快了约 49.9 倍，global-load 仍是 32 sector/request？

`(128,4096)` 的 V1 一行一个 block、每 block 256 线程，故每线程处理
`ceil(4096/256)=16` 列。初版列映射是 `tid×16+i`：对固定的循环轮次 `i`，
同一 warp 的 lane 0/1/2 分别读取第 0/16/32 列，地址间隔 64 字节，
所以 32 个 lane 各触及不同 sector。它只保证单个线程在时间上依次读取
连续数据，不保证同一条 warp load 指令的合并访存。

学习者 Profile 的 V1 global-load request 为 49152、sector 为 1572864，
`sector/request=32`，与 V0 完全相同；这支持上述地址分析。三轮 CUDA Event
Benchmark 中，`(128,4096)` V0/V1 平均为 `1.253107/0.025122 ms`，
延迟比约 `49.882×`。因此 V1 的确更快，但**不能**将该收益归因于已改善
的合并访存；V1 的更多 block 并行、每线程更短的串行循环等因素同时变化，
目前尚未量化各因素的独立贡献。

当时提出的验证方法是：调整线程到列的映射，让相邻 lane 在同一轮访问
相邻列，并在求 max、指数和、写输出三遍中保持一致；再以正确性、
Benchmark 和 `sector/request` 验证，而不是仅凭地址推断性能。NVIDIA 对
`sector/request` 的说明见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

### V1 列映射修正后，如何验证合并访存改善？

修正版每个线程从 `tid` 对应的列起步，每轮增加 `blockDim.x=256`。
固定一轮时，同一 warp 的相邻 lane 访问相邻 FP32 元素：32 个元素共
128 字节，若按 32 字节对齐，只覆盖 4 个 sector。学习者的
`(128,4096)` Nsight Compute 结果与推导一致：global-load request
保持 `49152`，sector 从初版 `1572864` 降到 `196608`，即从
`32` 降到 `4 sector/request`，sector 计数少了 `87.5%`。

17 个正确性用例与 Compute Sanitizer 均通过。相同构建配置下，三轮
CUDA Event 的 V1 延迟均值从 `0.025122 ms` 降到 `0.005178 ms`，
约快 `4.852×`。这两组证据分别证明读取侧请求更集中、整个 kernel
更快；但写回映射和编译后的指令路径也可能随之变化，不能把全部
`4.852×` 加速只归因于 global load sector 减少。sector 也不是实际
DRAM 读取量，ncu 的 `10.94 µs` 不与正常 Event 绝对时长混比。
本轮 V0 的第一轮延迟异常偏高，跨版本比较时不要机械地取该三轮均值。

### V1 的 `__syncthreads()` 分别保护什么？

每个线程先独立扫描自己负责的列，然后把局部 max 写入 `shared_max[tid]`；
第一次 barrier 保证整个 block 的局部值已经写好，其他线程才能开始读取
`shared_max[tid+stride]`。在 tree 的每一轮中，活跃线程更新一部分
shared 值；轮末 barrier 保证本轮写入对下一轮读取可见。特别是最后
`stride=1` 后也要让所有线程看到 `shared_max[0]` 的最终值。

指数和的 `shared_sum` 初始化及每轮 tree reduction 遵循同一规则。
`__syncthreads()` 只同步同一 block 的线程并建立这些访问的顺序，
不是数值正确性的自动检查；不能让部分线程在 barrier 前提前返回。
当前暂存实验里，写入和读回同一个 `output` 元素的是同一线程，
不需要为这对访问额外增设 block barrier。参见
[NVIDIA CUDA Programming Guide 的同步原语说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

### 先把 `expf(x-max)` 写到 output，少算一次指数会更快吗？

本次实验没有更快。`(128,4096)` 的三轮 Event 平均 latency 从直接重算版
`0.005178 ms` 增至暂存版 `0.009092 ms`，增加约 `75.6%`；其他三个
测试形状也均回退。源码层面，直接重算版每元素是三次 global load、
一次 global store、两次 `expf`；暂存版仍是三次 global load，只是第三遍
从 input 改为 output，并多一次 global store，`expf` 减为一次。
暂存版 global-load Profile 仍是 `49152 request / 196608 sector`，即
`4 sector/request`；这些指标没有测 store、缓存命中或实际 DRAM 流量，
所以不能把回退全部归因于某一个部件。两批 Benchmark 也不是同一轮
交替 A/B，百分比宜视为当前证据而非精确代价分解。减少源码中的计算
次数不保证 kernel 更快，必须看整个读、写、计算与同步路径。

### Softmax V1/V2 的 shared-load/store 指令数为什么能算到 34816/18432 与 2304/2304？

本次 Profile 有 `128` 个 block、每 block `256/32=8` 个 warp，max 和
sum 各规约一次。`smsp__inst_executed_op_shared_{ld,st}.sum` 统计的是
warp 级共享内存指令执行数，不能直接当成“线程读写元素数”或实际
shared-memory 事务数。对当前二进制，`cuobjdump` 可见 V1 每轮 tree
编译成两条带谓词的 `LDS` 和一条带谓词的 `STS`；ncu 的普通
`inst_executed` 会计入 warp 执行的这些指令，即便某些 lane 的谓词为假。

- V1 每次规约每 block 有 `8` 条初始化 store；8 轮各由 8 个 warp
  执行 `2 LDS + 1 STS`；最后 8 个 warp 各读一次行标量。所以两次
  规约的 load 为 `128×2×8×(8×2+1)=34816 inst`，store 为
  `128×2×8×(1+8)=18432 inst`。
- V2 每次规约每 block 有 8 个 warp partial store 和 1 个最终标量
  store，共 `9` 条 warp store；warp 0 读 partial 是 `1` 条 warp load，
  全 block 的 8 个 warp 读最终标量是 `8` 条，共 `9` 条 warp load。
  两次规约得到 `128×2×9=2304 inst`，load/store 都一样。

V1 `if (tid<stride)` 中实际参与的 warp 会随 stride 减少，不能把
上述 warp 指令数理解为所有 warp 每轮都进行了有效访存。若想验证
谓词影响，可用 ncu 查询 `smsp__inst_executed_op_shared_ld_pred_off_all.sum`
和对应 store 指标；它们与 `_pred_on_any` 区分整 warp 谓词关闭和
至少一个 lane 有效。NVIDIA 对 warp 级执行指令的定义见
[Nsight Compute 指标说明](https://docs.nvidia.com/nsight-compute/NsightCompute/)。

### V2 的 8 个 warp 都读同一个 shared 标量，广播后为什么仍是 8 条 load？

广播只解决**同一个 warp 内**多个 lane 读同一 shared 地址的服务问题：
该 warp 发出一条 load 指令，shared memory 可把同一个值提供给
参与的 lane，不因此产生 bank conflict。它不会替其他 warp 执行
指令。当前 block 有 8 个 warp，`row_max_shared` 或
`row_sum_shared` 由整个 block 读取时，各 warp 各发出一条 load，
因此 ncu 的 warp 指令计数是 `8`，不是 `1`，也不是按 256 个线程
计数。这个计数本身不等于物理请求次数或读取字节数。广播规则见
[NVIDIA CUDA shared-memory 访问说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。

再拆开 V1 的公式：`128 × 2 × 8 × (8×2+1)` 中，依次是
128 个 block、max/sum 两次规约、每 block 8 个 warp；括号内的
`8×2` 是 8 轮 tree 中每个 warp 每轮执行两条带谓词的 shared
load，`+1` 是最终读行标量。store 公式 `128 × 2 × 8 × (1+8)`
的 `1` 是每 warp 初始化局部值的一次 store，`8` 是 tree 的
8 轮各一次 store。以 stride=16 为例，只有 warp 0 的前 16 个
lane 真的操作数据；当前编译得到的指令指标仍可计入其他 warp
执行但所有 lane 谓词为假的 load/store。`inst_executed` 不等于
有效访存线程数。

### 为什么 Softmax V3 不能把每一行都直接当作 `float4*` 访问？

本节输入是行主序 FP32，行首相对分配基址的偏移为
`row * hidden * sizeof(float)` 字节。即使分配基址满足对齐，
`hidden` 不是 4 的倍数时，也可能只有部分行的行首满足 `float4`
所需的 16 字节对齐。V3 因而检查输入和输出行首的实际地址：
对齐行处理完整四元素组，不对齐行沿用标量路径；不足四个元素的
尾部也用标量处理。若基址已对齐，某行的条件是
`(row * hidden) % 4 == 0`；`hidden % 4 == 0` 只是在**所有行**
都要对齐时的充分条件，不是单独某行的必要条件。例如
`hidden=33` 时第 0、4、8…行可对齐，其余行不满足。
尾部若强行读取完整 `float4`，可能跨入下一行，最后一行还可能
越过整个分配范围。对齐和尾部处理保证访问合法，不证明一定更快。

### V3 的 global-load request 变成四分之一，为什么没有稳定加速？

本次 `(128,4096)` 的 V2/V3 一次 ncu 结果分别是
`49152/12288 request`，但两者都请求 `196608 sector`。
V2 每个 warp 每轮读 32 个 FP32，约为 4 个 32 字节 sector；
V3 每个 warp 每轮读 32 个 `float4`，约为 16 个 sector。
于是每次请求覆盖的 sector 数从 4 变 16，请求总数降四倍，
sector 总数不变。L1/TEX 的 request/sector 不等于 DRAM 读取量；
仅此两项也无法分离 `expf`、规约、对齐分支和其他指令的耗时。
三轮 Event 中 `(128,4096)` 平均为 V2 `0.006681 ms`、
V3 `0.006811 ms`，V3 约慢 `1.95%`；其他形状快慢不一，
所以应记录为“减少了读取请求，但未获得一致的整体性能收益”。

### FP16 Softmax 为什么先量化输入再生成 CPU Reference，规约却用 FP32？

本次 GPU 输入实际存储为 `__half`。若 CPU Reference 直接用尚未量化的
FP32 初始化值，比较结果会混入输入舍入误差，无法只检验 kernel。
因此测试夹具先将每个输入舍入为 half，再将其转回 float 供 double
CPU Reference 计算。核内 max、`expf`、指数和与归一化用 FP32，
避免在较长行里把每步累加都舍入到 FP16；最终输出才转换为 half。
`half2` 在这一练习里只改变成对读取/写回，不把 `__hadd2` 当作
Softmax 规约。标量 FP16 与 half2 保持相同 dtype 和算术精度，
才能较干净地比较访存分组变化；是否更快仍需 Event/Profile 证据。

### 为什么 half2 Softmax 的奇数列会失败，却有个别奇数形状仍显示 PASS？

本次初版在对齐行的指数和尾部循环里把 `expf(x-row_max)` 加进了
`thread_max`，随后参与 sum 规约的却是 `thread_sum`。这样分母
漏掉最后一个未配对的 half：`hidden=1` 时没有任何完整 pair，
`thread_sum=0`，归一化无效；其他奇数列的行和会偏离 1。
某个测试仍 PASS，只能说明漏掉的尾部贡献在该输入下未超过当前
误差阈值，不能证明尾部逻辑正确。先核对 max/sum/write 三个阶段的
尾部各自更新了正确变量，再复测正确性与内存安全，而不是放宽精度。

### FP16 Softmax 的 max/sum 为什么仍用 FP32？能改成 FP16 吗？

本次同 dtype 的标量/half2 对照只改变访存分组，保留相同 FP32
算术，便于判断成对访问的影响。FP16 输入已被量化，转成 FP32
不会恢复丢失的输入精度；但 `expf`、指数和、归一化用 FP32
能避免长行中每一步累加都再舍入到 FP16。max 只是从这些已量化
输入里选择最大值，单就“选出最大数”而言不必须用 FP32；保持
FP32 也方便与后续 FP32 规约统一。改成 FP16 数学运算并非禁止，
但会同时改变精度和性能，是另一组需要独立正确性与 Benchmark
验证的实验，不属于本次 half2 访存对照。

### `hidden=33` 时哪些行能用 half2？末尾一个 half 怎么处理？

一个 half 占 2 字节，half2 要求 4 字节对齐。若输入和输出基址
均对齐，则第 `row` 行首相对基址偏移 `row×33×2=66×row`
字节；偶数行满足 4 字节对齐，奇数行不满足。偶数行的前 32 个
元素可以分成 16 对，最后 1 个由标量路径处理；奇数行整行
使用标量路径。实际代码检查的是输入、输出行首的真实地址，
不能只看 `hidden` 推断所有行都走同一条路径。

### half2 的 request 减半，为什么 Softmax 没有稳定变快？

学习者在 `(128,4096)` 的一次 ncu 对照中测得，标量/half2
global-load request 为 `49152/24576`，sector 均为 `98304`；
即每 request 覆盖 sector 从 2 变成 4，总覆盖量没减少。
三轮 Event 中，`(128,4096)` 的平均延迟从 `0.006536 ms`
降到 `0.006190 ms`，约快 5.30%；但 `(128,1024)` 从
`0.003985 ms` 增到 `0.004212 ms`，约慢 5.70%。这说明
“减少访存指令请求”不等于“减少实际搬运量”，也不保证完整
kernel 加速。两版仍有 FP32 `expf`、两级规约、同步和数据转换；
当前指标不能分离它们的具体耗时贡献，更不能把 L1/TEX sector
直接当作 DRAM 流量。

## 10. Nsight Systems 时间线

### 服务器没有桌面，能用 Nsight Streamer 查看 `.nsys-rep` 吗？

可以。它在服务器的容器中运行 Nsight Systems GUI，通过 WebRTC
把界面传到浏览器；将已有报告目录挂载进容器后，即可在 GUI 中打开
报告，不需要把大文件下载到本地。单机可直接使用官方 Docker 镜像。

需要 Docker 及容器运行权限，还要让浏览器连通 HTTP 和 WebRTC/TURN
端口；当前官方单机默认端口为 TCP 8080、3478。使用 SSH 隧道时，
两个端口都要转发，实际连接仍需验证。GUI 应与生成报告的 CLI
版本相同或更新；采集程序无需在 Streamer 容器中运行。

当前官方镜像支持 `ENCODER=vp9` 软件编码；AV1 GPU 编码加速要求
Ada 或更新架构。因此本机 RTX 2080 Ti 可以选择软件模式。当前终端
PATH 中未找到 Docker/Podman，不能把文档支持视为部署已经成功。
报告较小时也可下载到本地 GUI 查看；纯终端先用
`nsys stats` 的事件报表辅助分析，但报表本身不提供交互式时间线。

来源：[NVIDIA NGC Nsight Streamer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys)、
[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

### 已在容器内，而且只能操作当前容器，怎么查看报告？

先区分当前容器和宿主机。容器内没有 Docker 命令，不代表宿主机
没有 Docker；要按官方 Docker 方式启动 Streamer，还需要能访问
Docker daemon。当前容器既没有 Docker socket，也未配置远程
Docker 地址，因此目前没有可直接使用的启动入口。

其他容器也不会自动看到当前容器里的报告。Docker bind mount 的
源路径属于 daemon 所在宿主机，不能直接把当前容器路径当作
宿主机路径，需要共享存储或先复制报告。

当前报告各约 150 KiB，建议下载到本地 Nsight Systems GUI。
若需要浏览器界面，可在当前容器内配置虚拟显示（如 Xvfb）、
VNC server 和 noVNC，再运行已有的 `nsys-ui`；这一路线无需
创建新容器，但需要安装相关依赖，并确认端口能经 SSH 隧道或
平台转发从浏览器访问。目前该路线尚未部署验证。

来源：[Docker bind mount 文档](https://docs.docker.com/engine/storage/bind-mounts/)、
[noVNC 官方说明](https://github.com/novnc/noVNC)。
