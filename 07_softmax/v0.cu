#include "softmax_harness.h"

// V0：一个线程处理一整行。先保证数值稳定与正确性，不做并行规约。
__global__ void softmax_v0_kernel(const float* input, float* output,
                                  int rows, int hidden) {
    // 一个线程负责一行；每行在行主序内存中连续存放。
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    // 尾部线程可能没有对应的行，必须先检查边界。
    if (row < rows) {
        const float* input_row = input + row * hidden;
        float* output_row = output + row * hidden;

        // 先减去本行最大值，避免大正数输入使 expf 溢出。
        float row_max = -INFINITY;
        for (int i = 0; i < hidden; i++) row_max = fmaxf(row_max, input_row[i]);

        // 串行累加分母；V0 为建立基线，多遍读取同一行输入。
        float row_sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            row_sum += expf(input_row[i] - row_max);
        }

        // 将每个元素的指数值除以行指数和。
        float inv_sum = 1.0f / row_sum;
        for (int i = 0; i < hidden; i++) {
            output_row[i] = expf(input_row[i] - row_max) * inv_sum;
        }
    }
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
