#include "softmax_final_harness.h"

// TODO：独立设计 warp 内 shuffle 和跨 warp 交接，不复制旧实现。
__global__ void softmax_final_warp_kernel(const float* input, float* output,
                                          int rows, int hidden) {
    // TODO：实现优化版二；自行添加需要的 warp helper，并验证 mask 的条件。
    // 以下只用于使空框架可编译；实现后删除这些占位语句。
    (void)input;
    (void)output;
    (void)rows;
    (void)hidden;
}

SoftmaxLaunchConfig make_softmax_config(int rows, int hidden) {
    // TODO：按你选择的映射计算 grid/block；下面不是最终启动配置。
    (void)rows;
    (void)hidden;
    return SoftmaxLaunchConfig{dim3(1), dim3(256)};
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
