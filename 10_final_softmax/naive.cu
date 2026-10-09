#include "softmax_final_harness.h"

// TODO：独立设计朴素版的线程映射、行处理方式和边界保护。
__global__ void softmax_final_naive_kernel(const float* input, float* output,
                                           int rows, int hidden) {
    // TODO：实现稳定的逐行 FP32 Softmax，不复制旧完整 kernel。
    // 以下只用于使空框架可编译；实现后删除这些占位语句。
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        const float* input_row = input + row * hidden;
        float* output_row = output + row * hidden;

        float row_max = -INFINITY;
        for (int i = 0; i < hidden; i++) {
            row_max = fmaxf(row_max, input_row[i]);
        }

        float row_sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            row_sum += expf(input_row[i] - row_max);
        }

        float inv_sum = 1.0f / row_sum;
        for (int i = 0; i < hidden; i++) {
            output_row[i] = expf(input_row[i] - row_max) * inv_sum;
        }
    }
}

SoftmaxLaunchConfig make_softmax_config(int rows, int hidden) {
    // TODO：按你选择的映射计算 grid/block；下面不是最终启动配置。
    dim3 block(128);
    dim3 grid((rows + block.x - 1) / block.x);
    return SoftmaxLaunchConfig{grid, block};
}

// input/output 均为已分配的 GPU 地址；分配、复制和同步由测试支撑负责。
void softmax_cuda(const float* input, float* output, int rows, int hidden) {
    const SoftmaxLaunchConfig config = make_softmax_config(rows, hidden);
    softmax_final_naive_kernel<<<config.grid, config.block>>>(
        input, output, rows, hidden);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    return run_final_softmax("final_naive", make_softmax_config,
                             softmax_cuda, argc, argv);
}
