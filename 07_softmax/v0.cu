#include "softmax_harness.h"

// V0：一个线程处理一整行。先保证数值稳定与正确性，不做并行规约。
__global__ void softmax_v0_kernel(const float* input, float* output,
                                  int rows, int hidden) {
    // TODO 1：用 blockIdx.x、blockDim.x、threadIdx.x 求出负责的行号。
    // TODO 2：检查行号是否越界；每行元素在 row-major 内存中连续存放。
    // TODO 3：先求这一行的最大值，再累加 exp(x - max) 得到分母。
    // TODO 4：计算每个输出的 exp(x - max) / 分母。
    // 注意：V0 允许多遍读取同一行；暂时不要加 shared memory 或 shuffle。
}

int main(int argc, char** argv) {
    const auto make_config = [](int rows, int /* hidden */) {
        const dim3 block(128);
        const dim3 grid((rows + static_cast<int>(block.x) - 1)
                        / static_cast<int>(block.x));
        return SoftmaxLaunchConfig{grid, block};
    };
    const auto launch = [](const float* input, float* output,
                           int rows, int hidden, SoftmaxLaunchConfig config) {
        softmax_v0_kernel<<<config.grid, config.block>>>(
            input, output, rows, hidden);
    };
    return run_softmax_exercise("v0_serial_row", make_config,
                                launch, argc, argv);
}
