#include <cstdint>

#include "softmax_harness.h"

constexpr int kSoftmaxV3BlockSize = 256;
constexpr int kSoftmaxV3WarpSize = 32;
constexpr int kSoftmaxV3VectorWidth = 4;
constexpr int kSoftmaxV3WarpsPerBlock =
    kSoftmaxV3BlockSize / kSoftmaxV3WarpSize;
constexpr unsigned int kSoftmaxV3FullWarpMask = 0xFFFFFFFFU;
static_assert(kSoftmaxV3BlockSize % kSoftmaxV3WarpSize == 0,
              "block 大小必须是 warp 大小的整数倍");

// 沿用 V2 已验收的两级规约，只把逐行数据访问改为 V3 的练习重点。
__device__ float warp_reduce_max_v3(float value) {
    for (int delta = kSoftmaxV3WarpSize / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value,
                      __shfl_down_sync(kSoftmaxV3FullWarpMask, value, delta));
    }
    return value;
}

__device__ float warp_reduce_sum_v3(float value) {
    for (int delta = kSoftmaxV3WarpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(kSoftmaxV3FullWarpMask, value, delta);
    }
    return value;
}

// 一行一个 block，输入/输出均为行主序 FP32 [rows, hidden]。
__global__ void softmax_v3_kernel(const float* input, float* output,
                                  int rows, int hidden) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % kSoftmaxV3WarpSize;
    const int warp = tid / kSoftmaxV3WarpSize;
    const int row_start = row * hidden;
    const float* input_row = input + row_start;
    float* output_row = output + row_start;

    // float4 访问要求实际行首地址满足 float4 对齐；不能只检查 hidden。
    // 非对齐行沿用 V2 的标量访问，不让重解释指针造成未对齐访问。
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(input_row) % alignof(float4) == 0 &&
        reinterpret_cast<std::uintptr_t>(output_row) % alignof(float4) == 0;
    const float4* input_row4 = reinterpret_cast<const float4*>(input_row);
    float4* output_row4 = reinterpret_cast<float4*>(output_row);

    const int vector_groups = hidden / kSoftmaxV3VectorWidth;
    const int tail_start = vector_groups * kSoftmaxV3VectorWidth;

    __shared__ float warp_max[kSoftmaxV3WarpsPerBlock];
    __shared__ float warp_sum[kSoftmaxV3WarpsPerBlock];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    float thread_max = -INFINITY;
    if (aligned) {
        // 相邻线程处理相邻的四元素组，后续组按 block 大小跨步。
        for (int group = tid; group < vector_groups;
             group += kSoftmaxV3BlockSize) {
            const float4 v = input_row4[group];
            thread_max = fmaxf(thread_max, v.x);
            thread_max = fmaxf(thread_max, v.y);
            thread_max = fmaxf(thread_max, v.z);
            thread_max = fmaxf(thread_max, v.w);
        }
        // 尾部不足四元素，以标量方式并入局部最大值。
        for (int col = tail_start + tid; col < hidden;
             col += kSoftmaxV3BlockSize) {
            thread_max = fmaxf(thread_max, input_row[col]);
        }
    } else {
        for (int col = tid; col < hidden; col += kSoftmaxV3BlockSize) {
            thread_max = fmaxf(thread_max, input_row[col]);
        }
    }

    const float local_warp_max = warp_reduce_max_v3(thread_max);
    if (lane == 0)
        warp_max[warp] = local_warp_max;
    __syncthreads();
    if (warp == 0) {
        const float partial = lane < kSoftmaxV3WarpsPerBlock
                                  ? warp_max[lane]
                                  : -INFINITY;
        const float max_val = warp_reduce_max_v3(partial);
        if (lane == 0)
            row_max_shared = max_val;
    }
    __syncthreads();
    const float row_max = row_max_shared;

    float thread_sum = 0.0f;
    if (aligned) {
        // 保持与 max 阶段相同的组映射，逐分量计算指数并累加。
        for (int group = tid; group < vector_groups;
             group += kSoftmaxV3BlockSize) {
            const float4 v = input_row4[group];
            thread_sum += expf(v.x - row_max);
            thread_sum += expf(v.y - row_max);
            thread_sum += expf(v.z - row_max);
            thread_sum += expf(v.w - row_max);
        }
        // 剩余列仍以标量方式参与分母计算。
        for (int col = tail_start + tid; col < hidden;
             col += kSoftmaxV3BlockSize) {
            thread_sum += expf(input_row[col] - row_max);
        }
    } else {
        for (int col = tid; col < hidden; col += kSoftmaxV3BlockSize) {
            thread_sum += expf(input_row[col] - row_max);
        }
    }

    const float local_warp_sum = warp_reduce_sum_v3(thread_sum);
    if (lane == 0)
        warp_sum[warp] = local_warp_sum;
    __syncthreads();
    if (warp == 0) {
        const float partial = lane < kSoftmaxV3WarpsPerBlock
                                  ? warp_sum[lane]
                                  : 0.0f;
        const float sum_val = warp_reduce_sum_v3(partial);
        if (lane == 0)
            row_sum_shared = sum_val;
    }
    __syncthreads();
    const float inv_sum = 1.0f / row_sum_shared;

    if (aligned) {
        // 每组一次向量化写回；四个分量仍各自计算 expf 与归一化。
        for (int group = tid; group < vector_groups;
             group += kSoftmaxV3BlockSize) {
            const float4 input_v = input_row4[group];
            float4 output_v;
            output_v.x = expf(input_v.x - row_max) * inv_sum;
            output_v.y = expf(input_v.y - row_max) * inv_sum;
            output_v.z = expf(input_v.z - row_max) * inv_sum;
            output_v.w = expf(input_v.w - row_max) * inv_sum;
            output_row4[group] = output_v;
        }
        // 尾部标量写回，避免完整 float4 跨过行边界。
        for (int col = tail_start + tid; col < hidden;
             col += kSoftmaxV3BlockSize) {
            output_row[col] = expf(input_row[col] - row_max) * inv_sum;
        }
    } else {
        for (int col = tid; col < hidden; col += kSoftmaxV3BlockSize) {
            output_row[col] = expf(input_row[col] - row_max) * inv_sum;
        }
    }

    // 即使线程没有负责的列，也不能在任一 shuffle/barrier 前提前返回。
    // 当前固定 256 个真实线程，8 个 warp 都完整参与 full-mask shuffle。
}

int main(int argc, char** argv) {
    const auto make_config = [](int rows, int /* hidden */) {
        return SoftmaxLaunchConfig{
            dim3(static_cast<unsigned int>(rows)),
            dim3(kSoftmaxV3BlockSize)};
    };
    const auto launch = [](const float* input, float* output,
                           int rows, int hidden, SoftmaxLaunchConfig config) {
        softmax_v3_kernel<<<config.grid, config.block>>>(
            input, output, rows, hidden);
    };
    return run_softmax_exercise("v3_float4", make_config,
                                launch, argc, argv);
}
