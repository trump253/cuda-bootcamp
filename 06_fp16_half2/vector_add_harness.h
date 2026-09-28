#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

// 这部分是三种版本共用的测试夹具，不是本节要重复编写的 kernel。
template <typename T>
struct NumberCodec;

template <>
struct NumberCodec<float> {
    static float from_float(float value) { return value; }
    static float to_float(float value) { return value; }
    static double tolerance() { return 1.0e-6; }
};

template <>
struct NumberCodec<__half> {
    static __half from_float(float value) { return __float2half_rn(value); }
    static float to_float(__half value) { return __half2float(value); }
    static double tolerance() { return 1.0e-3; }
};

template <typename T>
void initialize_inputs(std::vector<T>& a, std::vector<T>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float av = static_cast<float>(static_cast<int>(i % 37) - 18)
                         * 0.03125F + 0.1F;
        const float bv = static_cast<float>(static_cast<int>(i % 29) - 14)
                         * 0.015625F - 0.2F;
        a[i] = NumberCodec<T>::from_float(av);
        b[i] = NumberCodec<T>::from_float(bv);
    }
}

template <typename T>
void cpu_reference(const std::vector<T>& a,
                   const std::vector<T>& b,
                   std::vector<T>& expected) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float sum = NumberCodec<T>::to_float(a[i])
                          + NumberCodec<T>::to_float(b[i]);
        expected[i] = NumberCodec<T>::from_float(sum);
    }
}

inline dim3 make_grid(std::size_t n, unsigned int values_per_thread,
                      dim3 block) {
    const std::size_t work_items =
        (n + values_per_thread - 1) / values_per_thread;
    return dim3(static_cast<unsigned int>(
        (work_items + block.x - 1) / block.x));
}

template <typename T, typename Launch>
bool run_correctness_case(const char* variant, std::size_t n,
                          unsigned int values_per_thread,
                          dim3 block, Launch launch) {
    const std::size_t bytes = n * sizeof(T);
    std::vector<T> a(n), b(n), expected(n), actual(n);
    initialize_inputs(a, b);
    cpu_reference(a, b, expected);

    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    // NaN 哨兵：kernel 漏写的输出不能碰巧通过参考值检查。
    CUDA_CHECK(cudaMemset(d_c, 0xFF, bytes));

    const dim3 grid = make_grid(n, values_per_thread, block);
    launch(d_a, d_b, d_c, n, grid, block);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    double max_abs_error = 0.0;
    bool passed = true;
    for (std::size_t i = 0; i < n; ++i) {
        const double reference = NumberCodec<T>::to_float(expected[i]);
        const double result = NumberCodec<T>::to_float(actual[i]);
        if (!std::isfinite(reference) || !std::isfinite(result)) {
            passed = false;
            max_abs_error = std::numeric_limits<double>::infinity();
            continue;
        }
        const double error = std::fabs(reference - result);
        max_abs_error = std::max(max_abs_error, error);
        if (error > NumberCodec<T>::tolerance()) passed = false;
    }
    std::printf("variant=%s, N=%zu, grid=%u, block=%u, "
                "最大绝对误差=%.6g, %s\n",
                variant, n, grid.x, block.x, max_abs_error,
                passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return passed;
}

template <typename T, typename Launch>
void run_benchmark_case(const char* variant, std::size_t n,
                        unsigned int values_per_thread,
                        dim3 block, Launch launch) {
    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;
    const std::size_t bytes = n * sizeof(T);
    std::vector<T> a(n), b(n);
    initialize_inputs(a, b);

    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    const dim3 grid = make_grid(n, values_per_thread, block);

    for (int i = 0; i < kWarmup; ++i) {
        launch(d_a, d_b, d_c, n, grid, block);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        launch(d_a, d_b, d_c, n, grid, block);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double latency_ms = total_ms / kIterations;
    // 有效带宽按算法必须读 A、B、写 C 的字节数计，不等于实际 DRAM 流量。
    const double bandwidth_gbps =
        3.0 * static_cast<double>(bytes) / (latency_ms * 1.0e6);
    std::printf("variant=%s, N=%zu, grid=%u, block=%u, "
                "warmup=%d, iterations=%d, latency=%.6f ms, "
                "effective_bandwidth=%.2f GB/s\n",
                variant, n, grid.x, block.x, kWarmup, kIterations,
                latency_ms, bandwidth_gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

template <typename T, typename Launch>
int run_vector_add_exercise(const char* variant,
                            unsigned int values_per_thread,
                            Launch launch, int argc, char** argv) {
    const dim3 block(256);
    if (argc == 2 && std::strcmp(argv[1], "--profile") == 0) {
        // 只启动一次大尺寸 kernel，便于 Nsight Compute 精确选中目标。
        constexpr std::size_t n = 1U << 24;
        const std::size_t bytes = n * sizeof(T);
        std::vector<T> a(n), b(n);
        initialize_inputs(a, b);
        T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        const dim3 grid = make_grid(n, values_per_thread, block);
        launch(d_a, d_b, d_c, n, grid, block);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("profile variant=%s, N=%zu, grid=%u, block=%u\n",
                    variant, n, grid.x, block.x);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        return 0;
    }
    const bool correctness_only =
        argc == 2 && std::strcmp(argv[1], "--correctness-only") == 0;
    if (argc != 1 && !correctness_only) {
        std::fprintf(stderr, "用法：%s [--correctness-only|--profile]\n", argv[0]);
        return 2;
    }

    const std::size_t test_sizes[] = {
        1, 2, 3, 31, 32, 33, 255, 256, 257, 1000, (1U << 20) + 3U};
    bool all_passed = true;
    for (std::size_t n : test_sizes) {
        all_passed = run_correctness_case<T>(
            variant, n, values_per_thread, block, launch) && all_passed;
    }
    if (!all_passed) return 1;
    if (!correctness_only) {
        const std::size_t benchmark_sizes[] = {1024, 1U << 20, 1U << 24};
        for (std::size_t n : benchmark_sizes) {
            run_benchmark_case<T>(
                variant, n, values_per_thread, block, launch);
        }
    }
    return 0;
}
