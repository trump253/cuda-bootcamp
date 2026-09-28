#include <cuda_runtime.h>

#include "cuda_check.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

constexpr int kBlockSize = 256;

__global__ void reduce_sum_v0_kernel(const float* input,
                                     float* output,
                                     std::size_t n) {
    // V0 只让全局线程 0 工作，由它串行累加全部输入并写入 output[0]。
    // 累加变量在线程内部初始化，不依赖 output 的原始内容。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; i++) {
        sum += input[i];
    }
    output[0] = sum;
}

double reduce_sum_cpu(const std::vector<float>& input) {
    double sum = 0.0;
    for (const float value : input) {
        sum += static_cast<double>(value);
    }
    return sum;
}

void initialize_input(std::vector<float>& input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        // 使用可精确表示的正数，避免数据相互抵消掩盖漏算问题。
        input[i] = 1.0F + static_cast<float>(i % 7) * 0.125F;
    }
}

bool run_correctness_case(std::size_t n, int block_size) {
    if (n == 0 || block_size <= 0) {
        return false;
    }

    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n);
    initialize_input(input);
    const double expected = reduce_sum_cpu(input);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    // V0 故意只启动一个 block，并由全局线程 0 串行完成 reduction。
    reduce_sum_v0_kernel<<<1, block_size>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float actual = std::numeric_limits<float>::quiet_NaN();
    CUDA_CHECK(cudaMemcpy(&actual, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    const double abs_error =
        std::fabs(expected - static_cast<double>(actual));
    constexpr double kTolerance = 1e-5;
    const bool passed = std::isfinite(actual) && abs_error <= kTolerance;
    std::printf(
        "variant=v0_serial, N=%zu, grid=1, block=%d, expected=%.6f, "
        "actual=%.6f, abs_error=%.6g, %s\n",
        n, block_size, expected, static_cast<double>(actual), abs_error,
        passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return passed;
}

float benchmark_reduce_sum_v0(const float* d_input,
                              float* d_output,
                              std::size_t n,
                              int block_size,
                              int warmup,
                              int iterations) {
    if (n == 0 || block_size <= 0 || warmup < 0 || iterations <= 0) {
        return std::numeric_limits<float>::infinity();
    }

    for (int i = 0; i < warmup; ++i) {
        reduce_sum_v0_kernel<<<1, block_size>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        reduce_sum_v0_kernel<<<1, block_size>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(iterations);
}

bool run_benchmark_case(std::size_t n,
                        int block_size,
                        int warmup,
                        int iterations) {
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n);
    initialize_input(input);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    const float latency_ms = benchmark_reduce_sum_v0(
        d_input, d_output, n, block_size, warmup, iterations);
    // Reduction 的有效带宽只计算主要输入流量；单个标量输出可以忽略。
    const double effective_bandwidth_gbps =
        static_cast<double>(bytes) / (static_cast<double>(latency_ms) * 1.0e6);
    std::printf(
        "variant=v0_serial, N=%zu, grid=1, block=%d, warmup=%d, "
        "iterations=%d, latency=%.6f ms, effective_bandwidth=%.2f GB/s\n",
        n, block_size, warmup, iterations, latency_ms,
        effective_bandwidth_gbps);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return std::isfinite(latency_ms) && latency_ms > 0.0F &&
           std::isfinite(effective_bandwidth_gbps) &&
           effective_bandwidth_gbps > 0.0;
}

int main() {
    const std::vector<std::size_t> correctness_sizes = {
        1,
        31,
        32,
        33,
        255,
        256,
        257,
        1000,
        (1U << 20) + 3,
    };

    bool all_correct = true;
    for (const std::size_t n : correctness_sizes) {
        all_correct = run_correctness_case(n, kBlockSize) && all_correct;
    }
    if (!all_correct) {
        return EXIT_FAILURE;
    }

    constexpr int kWarmup = 2;
    constexpr int kIterations = 10;
    const std::vector<std::size_t> benchmark_sizes = {
        1U << 10,
        1U << 14,
        1U << 18,
        1U << 20,
    };

    bool all_benchmarks_valid = true;
    for (const std::size_t n : benchmark_sizes) {
        all_benchmarks_valid =
            run_benchmark_case(n, kBlockSize, kWarmup, kIterations) &&
            all_benchmarks_valid;
    }
    return all_benchmarks_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
