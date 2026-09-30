#include <cstdint>

#include "softmax_fp16_harness.h"

#ifndef SOFTMAX_FP16_USE_HALF2
#define SOFTMAX_FP16_USE_HALF2 0
#endif

constexpr int kSoftmaxFp16BlockSize = 256;
constexpr int kSoftmaxFp16WarpSize = 32;
constexpr int kSoftmaxFp16WarpsPerBlock =
    kSoftmaxFp16BlockSize / kSoftmaxFp16WarpSize;
constexpr unsigned int kSoftmaxFp16FullWarpMask = 0xFFFFFFFFU;
static_assert(alignof(__half2) == 4, "本练习假设 half2 需要 4 字节对齐");

// 与 V2 相同的两级规约；FP16 仅是输入/输出存储格式，规约值保持 FP32。
__device__ float warp_reduce_max_fp16(float value) {
    for (int delta = kSoftmaxFp16WarpSize / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value,
                      __shfl_down_sync(kSoftmaxFp16FullWarpMask, value, delta));
    }
    return value;
}

__device__ float warp_reduce_sum_fp16(float value) {
    for (int delta = kSoftmaxFp16WarpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(kSoftmaxFp16FullWarpMask, value, delta);
    }
    return value;
}

// 同一源文件生成标量 FP16 基线与 half2 练习目标；唯一变量是访存分组。
__global__ void softmax_fp16_kernel(const __half* input, __half* output,
                                    int rows, int hidden) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % kSoftmaxFp16WarpSize;
    const int warp = tid / kSoftmaxFp16WarpSize;
    const int row_start = row * hidden;
    const __half* input_row = input + row_start;
    __half* output_row = output + row_start;

#if SOFTMAX_FP16_USE_HALF2
    // 只有输入、输出行首都满足 half2 对齐时，才访问完整的 half 对。
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(input_row) % alignof(__half2) == 0 &&
        reinterpret_cast<std::uintptr_t>(output_row) % alignof(__half2) == 0;
    const int pair_count = hidden / 2;
    const int tail_start = pair_count * 2;
    const __half2* input_row2 = reinterpret_cast<const __half2*>(input_row);
    __half2* output_row2 = reinterpret_cast<__half2*>(output_row);
#endif

    __shared__ float warp_max[kSoftmaxFp16WarpsPerBlock];
    __shared__ float warp_sum[kSoftmaxFp16WarpsPerBlock];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    float thread_max = -INFINITY;
#if SOFTMAX_FP16_USE_HALF2
    if (aligned) {
        // TODO 1：每个线程读取 pair=tid、tid+256、... 的两个 half，
        // 分别提升为 float 后更新 thread_max。
        for (int pair = tid; pair < pair_count;
             pair += kSoftmaxFp16BlockSize) {
            // TODO：读取一个 half2，并将两个分量纳入 max。
            const float2 f2 = __half22float2(input_row2[pair]);
            thread_max = fmaxf(thread_max, f2.x);
            thread_max = fmaxf(thread_max, f2.y);
        }
        // TODO 2：处理不足一对的标量尾部。
        for (int col = tail_start + tid; col < hidden;
             col += kSoftmaxFp16BlockSize) {
            // TODO：将尾部 half 提升为 float 后纳入 max。
            thread_max = fmaxf(thread_max, __half2float(input_row[col]));
        }
    } else
#endif
    {
        for (int col = tid; col < hidden; col += kSoftmaxFp16BlockSize) {
            thread_max = fmaxf(thread_max, __half2float(input_row[col]));
        }
    }

    const float local_warp_max = warp_reduce_max_fp16(thread_max);
    if (lane == 0)
        warp_max[warp] = local_warp_max;
    __syncthreads();
    if (warp == 0) {
        const float partial = lane < kSoftmaxFp16WarpsPerBlock
                                  ? warp_max[lane]
                                  : -INFINITY;
        const float max_val = warp_reduce_max_fp16(partial);
        if (lane == 0)
            row_max_shared = max_val;
    }
    __syncthreads();
    const float row_max = row_max_shared;

    float thread_sum = 0.0f;
#if SOFTMAX_FP16_USE_HALF2
    if (aligned) {
        // TODO 3：成对读取，分别用 FP32 计算 expf(x-row_max) 并累加。
        for (int pair = tid; pair < pair_count;
             pair += kSoftmaxFp16BlockSize) {
            // TODO：读取 half2，将两个分量计入 thread_sum。
            const float2 f2 = __half22float2(input_row2[pair]);
            thread_sum += expf(f2.x - row_max);
            thread_sum += expf(f2.y - row_max);
        }
        // TODO 4：将尾部标量计入 thread_sum。
        for (int col = tail_start + tid; col < hidden;
             col += kSoftmaxFp16BlockSize) {
            // TODO：在这里处理尾部。
            thread_sum += expf(__half2float(input_row[col]) - row_max);
        }
    } else
#endif
    {
        for (int col = tid; col < hidden; col += kSoftmaxFp16BlockSize) {
            thread_sum += expf(__half2float(input_row[col]) - row_max);
        }
    }

    const float local_warp_sum = warp_reduce_sum_fp16(thread_sum);
    if (lane == 0)
        warp_sum[warp] = local_warp_sum;
    __syncthreads();
    if (warp == 0) {
        const float partial = lane < kSoftmaxFp16WarpsPerBlock
                                  ? warp_sum[lane]
                                  : 0.0f;
        const float sum_val = warp_reduce_sum_fp16(partial);
        if (lane == 0)
            row_sum_shared = sum_val;
    }
    __syncthreads();
    const float inv_sum = 1.0f / row_sum_shared;

#if SOFTMAX_FP16_USE_HALF2
    if (aligned) {
        // TODO 5：成对写回两个归一化结果；只在最终写回时舍入到 FP16。
        for (int pair = tid; pair < pair_count;
             pair += kSoftmaxFp16BlockSize) {
            // TODO：写回这一对 half 输出。
            const float2 f2 = __half22float2(input_row2[pair]);
            output_row2[pair] = __floats2half2_rn(expf(f2.x - row_max) * inv_sum,
                                                  expf(f2.y - row_max) * inv_sum);
        }
        // TODO 6：标量写回不足一对的尾部。
        for (int col = tail_start + tid; col < hidden;
             col += kSoftmaxFp16BlockSize) {
            // TODO：写回尾部 half 输出。
            const float value = expf(__half2float(input_row[col]) - row_max) * inv_sum;
            output_row[col] = __float2half_rn(value);
        }
    } else
#endif
    {
        for (int col = tid; col < hidden; col += kSoftmaxFp16BlockSize) {
            const float value =
                expf(__half2float(input_row[col]) - row_max) * inv_sum;
            output_row[col] = __float2half_rn(value);
        }
    }

    // 无有效元素的线程仍须参加 shuffle 和 block barrier；不能提前返回。
}

int main(int argc, char** argv) {
    const auto launch = [](const __half* input, __half* output,
                           int rows, int hidden, SoftmaxLaunchConfig config) {
        softmax_fp16_kernel<<<config.grid, config.block>>>(
            input, output, rows, hidden);
    };
#if SOFTMAX_FP16_USE_HALF2
    return run_softmax_fp16_exercise("fp16_half2", launch, argc, argv);
#else
    return run_softmax_fp16_exercise("fp16_scalar", launch, argc, argv);
#endif
}
