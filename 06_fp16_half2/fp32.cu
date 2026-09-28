#include "vector_add_harness.h"

// FP32 对照组：复用 Day 1 的一线程一元素映射。
__global__ void vector_add_fp32_kernel(const float* a, const float* b,
                                       float* c, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    const auto launch = [](const float* a, const float* b, float* c,
                           std::size_t n, dim3 grid, dim3 block) {
        vector_add_fp32_kernel<<<grid, block>>>(a, b, c, n);
    };
    return run_vector_add_exercise<float>("fp32", 1, launch, argc, argv);
}
