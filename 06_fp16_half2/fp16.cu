#include "vector_add_harness.h"

// 你来完成：每个线程处理一个 __half 元素。
__global__ void vector_add_fp16_kernel(const __half* a, const __half* b,
                                       __half* c, std::size_t n) {
    // TODO：计算一维全局索引并检查 i < n。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO：使用 __hadd 对两个 half 做一次 FP16 加法并写回。
    // 这样与 half2 版保持相同数值路径，主要比较一线程处理 1 个或 2 个元素。
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

int main(int argc, char** argv) {
    const auto launch = [](const __half* a, const __half* b, __half* c,
                           std::size_t n, dim3 grid, dim3 block) {
        vector_add_fp16_kernel<<<grid, block>>>(a, b, c, n);
    };
    return run_vector_add_exercise<__half>("fp16_scalar", 1,
                                           launch, argc, argv);
}
