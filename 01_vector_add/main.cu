#include <cuda_runtime.h>

#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

__global__ void hello_kernel() {
    // 输出 block、block 内线程及其对应的全局线程编号。
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = bid * blockDim.x + tid;
    printf("blockIdx.x: %d, threadIdx.x: %d, globalIdx.x: %d\n", bid, tid, gid);
}

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  std::size_t n) {
    // 将当前线程映射到一个元素，并对末尾的额外线程进行边界保护。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      c[idx] = a[idx] + b[idx];
    }
}

void vector_add_cpu(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c) {
    // 计算 CPU 端的参考结果。
    for (int i = 0; i < a.size(); i++) {
      c[i] = a[i] + b[i];
    }
}

float max_abs_error(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        return std::numeric_limits<float>::infinity();
    }

    float max_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(expected[i] - actual[i]);
        if (!std::isfinite(error)) {
            return std::numeric_limits<float>::infinity();
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

bool run_case(std::size_t n, int block_size) {
    // 为一组输入执行 CPU 参考计算、GPU 计算和结果验证。
    constexpr float kTolerance = 1e-6F;
    const std::size_t size = n * sizeof(float);

    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_C(n, 0.0F);
    std::vector<float> expected(n);

    for (std::size_t i = 0; i < n; ++i) {
      h_A[i] = static_cast<float>(i);
      h_B[i] = static_cast<float>(i * 2);
    }
    vector_add_cpu(h_A, h_B, expected);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    vector_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    const float error = max_abs_error(expected, h_C);
    const bool passed = error <= kTolerance;
    std::cout << "N=" << n
              << ", grid=" << grid_size
              << ", block=" << block_size
              << ", 最大绝对误差=" << error
              << ", 结果=" << (passed ? "通过" : "失败") << '\n';
    return passed;
}

int main() {
    constexpr int kBlockSize = 256;
    const std::vector<std::size_t> test_sizes = {
        1, 255, 256, 257, 1000, (1U << 20) + 3};

    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool all_passed = true;
    for (const std::size_t n : test_sizes) {
        all_passed = run_case(n, kBlockSize) && all_passed;
    }

    std::cout << "Day 1 验证结果："
              << (all_passed ? "全部通过" : "存在失败") << '\n';
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
