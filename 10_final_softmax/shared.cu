#include "softmax_final_harness.h"

constexpr int SoftmaxBlockSize = 256;

// TODO：独立设计 block 内 shared reduction 及其同步，不复制旧实现。
__global__ void softmax_final_shared_kernel(const float* input, float* output,
                                            int rows, int hidden) {
    // TODO：实现优化版一；说明 shared 数据的用途、生命周期和边界处理。
    // 以下只用于使空框架可编译；实现后删除这些占位语句。
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float shared_max[SoftmaxBlockSize];
    __shared__ float shared_sum[SoftmaxBlockSize];

    int row_start = row * hidden;
    float thread_max = -INFINITY;
    for (int i = tid; i < hidden; i += SoftmaxBlockSize) {
        thread_max = fmaxf(thread_max, input[row_start + i]);
    }
    shared_max[tid] = thread_max;
    __syncthreads();

    for (int stride = SoftmaxBlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared_max[0];

    float thread_sum = 0.0f;
    for (int i = tid; i < hidden; i += SoftmaxBlockSize) {
        thread_sum += expf(input[row_start + i] - row_max);
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int stride = SoftmaxBlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / shared_sum[0];
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
    softmax_final_shared_kernel<<<config.grid, config.block>>>(
        input, output, rows, hidden);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_final_softmax("final_shared", make_softmax_config,
                             softmax_cuda, argc, argv);
}
