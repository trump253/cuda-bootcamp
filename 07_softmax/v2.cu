#include "softmax_harness.h"

constexpr int kSoftmaxV2BlockSize = 256;
constexpr int kSoftmaxV2WarpSize = 32;
constexpr int kSoftmaxV2WarpsPerBlock =
    kSoftmaxV2BlockSize / kSoftmaxV2WarpSize;
constexpr unsigned int kSoftmaxV2FullWarpMask = 0xFFFFFFFFU;
static_assert(kSoftmaxV2BlockSize % kSoftmaxV2WarpSize == 0,
              "block 大小必须是 warp 大小的整数倍");

__device__ float warp_reduce_max_v2(float value) {
    // 32 个真实 lane 都参与 shuffle；规约结果由 lane 0 使用。
    for (int delta = kSoftmaxV2WarpSize / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kSoftmaxV2FullWarpMask, value, delta));
    }
    return value;
}

__device__ float warp_reduce_sum_v2(float value) {
    // 与 max 相同，逐步合并距离 delta 处的 lane 值。
    for (int delta = kSoftmaxV2WarpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(kSoftmaxV2FullWarpMask, value, delta);
    }
    return value;
}

// V2：一个 block 处理一行；列映射、测试口径与已验收的 V1 保持一致。
__global__ void softmax_v2_kernel(const float* input, float* output,
                                  int rows, int hidden) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % kSoftmaxV2WarpSize;
    const int warp = tid / kSoftmaxV2WarpSize;
    const int row_start = row * hidden;

    __shared__ float warp_max[kSoftmaxV2WarpsPerBlock];
    __shared__ float warp_sum[kSoftmaxV2WarpsPerBlock];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    float thread_max = -INFINITY;
    for (int col = tid; col < hidden; col += kSoftmaxV2BlockSize) {
        thread_max = fmaxf(thread_max, input[row_start + col]);
    }

    // 第一层：每个 warp 先规约自己的局部最大值，lane 0 写出 partial。
    const float local_warp_max = warp_reduce_max_v2(thread_max);
    if (lane == 0) {
        warp_max[warp] = local_warp_max;
    }
    __syncthreads();  // 等待全部 warp 的 partial max 写入。

    // warp 0 规约 8 个 partial；其余 24 个 lane 以 -INFINITY 补齐。
    if (warp == 0) {
        float max_val = warp_reduce_max_v2(lane < kSoftmaxV2WarpsPerBlock ? warp_max[lane] : -INFINITY);
        if (lane == 0) {
            row_max_shared = max_val;
        }
    }
    __syncthreads();  // 等待 warp 0 写出行最大值，再由整个 block 读取。
    float row_max = row_max_shared;

    float thread_sum = 0.0f;
    for (int col = tid; col < hidden; col += kSoftmaxV2BlockSize) {
        thread_sum += expf(input[row_start + col] - row_max);
    }

    // 第一层：每个 warp 先规约自己的局部指数和，lane 0 写出 partial。
    const float local_warp_sum = warp_reduce_sum_v2(thread_sum);
    if (lane == 0) {
        warp_sum[warp] = local_warp_sum;
    }
    __syncthreads();  // 等待全部 warp 的 partial sum 写入。

    // warp 0 规约 8 个 partial；其余 24 个 lane 以 0 补齐。
    if (warp == 0) {
        float sum_val = warp_reduce_sum_v2(lane < kSoftmaxV2WarpsPerBlock ? warp_sum[lane] : 0.0f);
        if (lane == 0) {
            row_sum_shared = sum_val;
        }
    }
    __syncthreads();  // 等待 warp 0 写出行指数和，再由整个 block 读取。
    float row_sum = row_sum_shared;

    const float inv_sum = 1.0f / row_sum;
    for (int col = tid; col < hidden; col += kSoftmaxV2BlockSize) {
        output[row_start + col] =
            expf(input[row_start + col] - row_max) * inv_sum;
    }

    // 注意：无有效列的线程也必须参加 warp shuffle 和 block barrier。
    // 当前固定 256 线程，所有 warp 都有 32 个真实 lane，才可使用 full mask。
}

int main(int argc, char** argv) {
    const auto make_config = [](int rows, int /* hidden */) {
        return SoftmaxLaunchConfig{
            dim3(static_cast<unsigned int>(rows)),
            dim3(kSoftmaxV2BlockSize)};
    };
    const auto launch = [](const float* input, float* output,
                           int rows, int hidden, SoftmaxLaunchConfig config) {
        softmax_v2_kernel<<<config.grid, config.block>>>(
            input, output, rows, hidden);
    };
    return run_softmax_exercise("v2_warp_shuffle", make_config,
                                launch, argc, argv);
}
