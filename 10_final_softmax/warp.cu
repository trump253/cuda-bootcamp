#include "softmax_final_harness.h"

constexpr int SoftmaxBlockSize = 256;
constexpr int WarpSize = 32;
constexpr int SoftmaxWarpsPerBlock = SoftmaxBlockSize / WarpSize;
constexpr unsigned int SoftmaxFullWarpMask = 0xFFFFFFFFU;

__device__ float warp_reduce_max(float value) {
    for (int delta = WarpSize / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down_sync(SoftmaxFullWarpMask, value, delta));
    }
    return value;
}

__device__ float warp_reduce_sum(float value) {
    for (int delta = WarpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(SoftmaxFullWarpMask, value, delta);
    }
    return value;
}

// TODO：独立设计 warp 内 shuffle 和跨 warp 交接，不复制旧实现。
__global__ void softmax_final_warp_kernel(const float* input, float* output,
                                          int rows, int hidden) {
    // TODO：实现优化版二；自行添加需要的 warp helper，并验证 mask 的条件。
    // 以下只用于使空框架可编译；实现后删除这些占位语句。
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WarpSize;
    int warp = tid / WarpSize;
    int row_start = row * hidden;

    __shared__ float shared_max[SoftmaxWarpsPerBlock];
    __shared__ float shared_sum[SoftmaxWarpsPerBlock];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    float thread_max = -INFINITY;
    for (int i = tid; i < hidden; i += SoftmaxBlockSize) {
        thread_max = fmaxf(thread_max, input[row_start + i]);
    }
    float warp_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        shared_max[warp] = warp_max;
    }
    __syncthreads();

    if (warp == 0) {
        float max_value = warp_reduce_max(lane < SoftmaxWarpsPerBlock ? shared_max[lane] : -INFINITY);
        if (lane == 0) {
            row_max_shared = max_value;
        }
    }
    __syncthreads();
    float row_max = row_max_shared;

    float thread_sum = 0.0f;
    for (int i = tid; i < hidden; i += SoftmaxBlockSize) {
        thread_sum += expf(input[row_start + i] - row_max);
    }
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        shared_sum[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float sum_value = warp_reduce_sum(lane < SoftmaxWarpsPerBlock ? shared_sum[lane] : 0.0f);
        if (lane == 0) {
            row_sum_shared = sum_value;
        }
    }
    __syncthreads();

    float inv_sum = 1.0f / row_sum_shared;
    for (int i = tid; i < hidden; i += SoftmaxBlockSize) {
        output[row_start + i] = expf(input[row_start + i] - row_max) * inv_sum;
    }
}

SoftmaxLaunchConfig make_softmax_config(int rows, int hidden) {
    // TODO：按你选择的映射计算 grid/block；下面不是最终启动配置。
    return SoftmaxLaunchConfig{dim3(rows), dim3(SoftmaxBlockSize)};
}

void softmax_cuda(const float* input, float* output, int rows, int hidden) {
    const SoftmaxLaunchConfig config = make_softmax_config(rows, hidden);
    softmax_final_warp_kernel<<<config.grid, config.block>>>(
        input, output, rows, hidden);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_final_softmax("final_warp", make_softmax_config,
                             softmax_cuda, argc, argv);
}
