#include "softmax_harness.h"

constexpr int kSoftmaxV1BlockSize = 256;

// V1：一个 block 协作处理一行；max 与 sum 都由 shared memory tree reduction 完成。
__global__ void softmax_v1_kernel(const float* input, float* output,
                                  int rows, int hidden) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float shared_max[kSoftmaxV1BlockSize];
    __shared__ float shared_sum[kSoftmaxV1BlockSize];

    // TODO 1：每个线程扫描本行由自己负责的列，得到局部最大值。
    //         无元素的线程必须给 max 规约提供正确的单位元。
    // TODO 2：所有线程协作，将局部最大值写入 shared_max 并规约出行最大值。
    // TODO 3：每个线程扫描负责的列，累加 expf(x - 行最大值)。
    //         无元素的线程必须给 sum 规约提供正确的单位元。
    // TODO 4：所有线程协作，将局部和写入 shared_sum 并规约出行指数和。
    // TODO 5：各线程将自己负责的列归一化并写入 output。
    // 注意：tid >= hidden 的线程不能提前返回；它仍需参与 block 内所有 barrier。
    // 注意：不同 block 的 shared memory 不共享；本 kernel 无需跨 block 同步。
}

int main(int argc, char** argv) {
    const auto make_config = [](int rows, int /* hidden */) {
        return SoftmaxLaunchConfig{
            dim3(static_cast<unsigned int>(rows)),
            dim3(kSoftmaxV1BlockSize)};
    };
    const auto launch = [](const float* input, float* output,
                           int rows, int hidden, SoftmaxLaunchConfig config) {
        softmax_v1_kernel<<<config.grid, config.block>>>(
            input, output, rows, hidden);
    };
    return run_softmax_exercise("v1_shared_tree", make_config,
                                launch, argc, argv);
}
