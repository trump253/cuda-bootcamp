#pragma once

#include <cuda_runtime.h>

#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

struct GemmShape {
    int m;
    int n;
    int k;
};

struct GemmErrorStats {
    double max_abs_error;
    double max_rel_error;
    bool all_finite;
    bool all_close;
};

inline std::size_t matrix_elements(int rows, int cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

inline void initialize_gemm_inputs(std::vector<float>& a,
                                   std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int value = static_cast<int>((i * 17U) % 29U) - 14;
        a[i] = static_cast<float>(value) * 0.03125F;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        const int value = static_cast<int>((i * 13U) % 31U) - 15;
        b[i] = static_cast<float>(value) * 0.025F;
    }
}

inline void gemm_cpu_reference(const std::vector<float>& a,
                               const std::vector<float>& b,
                               std::vector<float>& c,
                               int m,
                               int n,
                               int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double accumulator = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                accumulator +=
                    static_cast<double>(a[row * k + inner]) *
                    static_cast<double>(b[inner * n + col]);
            }
            c[row * n + col] = static_cast<float>(accumulator);
        }
    }
}

inline GemmErrorStats compare_gemm_results(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    double absolute_tolerance,
    double relative_tolerance) {
    GemmErrorStats stats{0.0, 0.0, true, true};
    if (expected.size() != actual.size()) {
        stats.max_abs_error = std::numeric_limits<double>::infinity();
        stats.max_rel_error = std::numeric_limits<double>::infinity();
        stats.all_finite = false;
        stats.all_close = false;
        return stats;
    }

    constexpr double kRelativeFloor = 1.0e-12;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double expected_value = static_cast<double>(expected[i]);
        const double actual_value = static_cast<double>(actual[i]);
        if (!std::isfinite(expected_value) || !std::isfinite(actual_value)) {
            stats.all_finite = false;
            stats.all_close = false;
            stats.max_abs_error = std::numeric_limits<double>::infinity();
            stats.max_rel_error = std::numeric_limits<double>::infinity();
            continue;
        }

        const double abs_error = std::fabs(expected_value - actual_value);
        const double rel_error =
            abs_error / std::max(std::fabs(expected_value), kRelativeFloor);
        stats.max_abs_error = std::max(stats.max_abs_error, abs_error);
        stats.max_rel_error = std::max(stats.max_rel_error, rel_error);
        if (abs_error > absolute_tolerance +
                            relative_tolerance * std::fabs(expected_value)) {
            stats.all_close = false;
        }
    }
    return stats;
}

template <typename LaunchGemm>
bool run_gemm_correctness_case(const char* variant,
                               GemmShape shape,
                               dim3 block,
                               LaunchGemm launch_gemm,
                               bool show_grid = true) {
    const std::size_t a_elements = matrix_elements(shape.m, shape.k);
    const std::size_t b_elements = matrix_elements(shape.k, shape.n);
    const std::size_t c_elements = matrix_elements(shape.m, shape.n);
    const std::size_t a_bytes = a_elements * sizeof(float);
    const std::size_t b_bytes = b_elements * sizeof(float);
    const std::size_t c_bytes = c_elements * sizeof(float);

    std::vector<float> h_a(a_elements);
    std::vector<float> h_b(b_elements);
    std::vector<float> expected(c_elements);
    std::vector<float> actual(c_elements,
                              std::numeric_limits<float>::quiet_NaN());
    initialize_gemm_inputs(h_a, h_b);
    gemm_cpu_reference(h_a, h_b, expected,
                       shape.m, shape.n, shape.k);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, c_bytes));

    launch_gemm(d_a, d_b, d_c, shape, block);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_c, c_bytes,
                          cudaMemcpyDeviceToHost));

    constexpr double kAbsoluteTolerance = 1.0e-3;
    constexpr double kRelativeTolerance = 1.0e-3;
    const GemmErrorStats stats = compare_gemm_results(
        expected, actual, kAbsoluteTolerance, kRelativeTolerance);
    const bool passed = stats.all_finite && stats.all_close;
    if (show_grid) {
        const dim3 grid(
            (shape.n + static_cast<int>(block.x) - 1) / block.x,
            (shape.m + static_cast<int>(block.y) - 1) / block.y);
        std::printf(
            "variant=%s, A=(%d,%d), B=(%d,%d), C=(%d,%d), "
            "block=(%u,%u), grid=(%u,%u), max_abs_error=%.6g, "
            "max_rel_error=%.6g, %s\n",
            variant,
            shape.m, shape.k, shape.k, shape.n, shape.m, shape.n,
            block.x, block.y, grid.x, grid.y,
            stats.max_abs_error, stats.max_rel_error,
            passed ? "PASS" : "FAIL");
    } else {
        std::printf(
            "variant=%s, A=(%d,%d), B=(%d,%d), C=(%d,%d), "
            "max_abs_error=%.6g, max_rel_error=%.6g, %s\n",
            variant,
            shape.m, shape.k, shape.k, shape.n, shape.m, shape.n,
            stats.max_abs_error, stats.max_rel_error,
            passed ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return passed;
}

template <typename LaunchGemm>
float benchmark_gemm(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     GemmShape shape,
                     dim3 block,
                     int warmup,
                     int iterations,
                     LaunchGemm launch_gemm) {
    for (int i = 0; i < warmup; ++i) {
        launch_gemm(d_a, d_b, d_c, shape, block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_gemm(d_a, d_b, d_c, shape, block);
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

template <typename LaunchGemm>
bool run_gemm_benchmark_case(const char* variant,
                             GemmShape shape,
                             dim3 block,
                             int warmup,
                             int iterations,
                             LaunchGemm launch_gemm,
                             bool show_grid = true) {
    const std::size_t a_elements = matrix_elements(shape.m, shape.k);
    const std::size_t b_elements = matrix_elements(shape.k, shape.n);
    const std::size_t c_elements = matrix_elements(shape.m, shape.n);
    const std::size_t a_bytes = a_elements * sizeof(float);
    const std::size_t b_bytes = b_elements * sizeof(float);
    const std::size_t c_bytes = c_elements * sizeof(float);

    std::vector<float> h_a(a_elements);
    std::vector<float> h_b(b_elements);
    initialize_gemm_inputs(h_a, h_b);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_bytes,
                          cudaMemcpyHostToDevice));

    const float latency_ms = benchmark_gemm(
        d_a, d_b, d_c, shape, block, warmup, iterations, launch_gemm);
    const double operations =
        2.0 * static_cast<double>(shape.m) *
        static_cast<double>(shape.n) * static_cast<double>(shape.k);
    const double performance_gflops =
        operations / (static_cast<double>(latency_ms) * 1.0e6);
    if (show_grid) {
        const dim3 grid(
            (shape.n + static_cast<int>(block.x) - 1) / block.x,
            (shape.m + static_cast<int>(block.y) - 1) / block.y);
        std::printf(
            "variant=%s, M=%d, N=%d, K=%d, block=(%u,%u), grid=(%u,%u), "
            "warmup=%d, iterations=%d, latency=%.6f ms, "
            "performance=%.2f GFLOP/s\n",
            variant, shape.m, shape.n, shape.k,
            block.x, block.y, grid.x, grid.y,
            warmup, iterations, static_cast<double>(latency_ms),
            performance_gflops);
    } else {
        std::printf(
            "variant=%s, M=%d, N=%d, K=%d, "
            "warmup=%d, iterations=%d, latency=%.6f ms, "
            "performance=%.2f GFLOP/s\n",
            variant, shape.m, shape.n, shape.k,
            warmup, iterations, static_cast<double>(latency_ms),
            performance_gflops);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return std::isfinite(latency_ms) && latency_ms > 0.0F &&
           std::isfinite(performance_gflops) && performance_gflops > 0.0;
}

template <typename LaunchGemm>
int run_gemm_profile_case(const char* variant,
                          dim3 block,
                          LaunchGemm launch_gemm) {
    const GemmShape shape{1024, 1024, 1024};
    const std::size_t a_bytes =
        matrix_elements(shape.m, shape.k) * sizeof(float);
    const std::size_t b_bytes =
        matrix_elements(shape.k, shape.n) * sizeof(float);
    const std::size_t c_bytes =
        matrix_elements(shape.m, shape.n) * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemset(d_a, 0, a_bytes));
    CUDA_CHECK(cudaMemset(d_b, 0, b_bytes));

    launch_gemm(d_a, d_b, d_c, shape, block);
    CUDA_CHECK(cudaDeviceSynchronize());
    const dim3 grid(
        (shape.n + static_cast<int>(block.x) - 1) / block.x,
        (shape.m + static_cast<int>(block.y) - 1) / block.y);
    std::printf(
        "profile variant=%s, M=%d, N=%d, K=%d, "
        "block=(%u,%u), grid=(%u,%u)\n",
        variant, shape.m, shape.n, shape.k,
        block.x, block.y, grid.x, grid.y);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return EXIT_SUCCESS;
}

template <typename LaunchGemm>
int run_gemm_program(int argc,
                     char** argv,
                     const char* variant,
                     dim3 block,
                     LaunchGemm launch_gemm) {
    if (argc == 2 && std::strcmp(argv[1], "--profile") == 0) {
        return run_gemm_profile_case(variant, block, launch_gemm);
    }
    if (argc != 1) {
        std::fprintf(stderr, "用法：%s [--profile]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::vector<GemmShape> correctness_shapes = {
        {1, 1, 1},
        {3, 5, 7},
        {31, 33, 17},
        {33, 31, 65},
        {64, 48, 37},
    };
    bool all_pass = true;
    for (const GemmShape shape : correctness_shapes) {
        all_pass = run_gemm_correctness_case(
                       variant, shape, block, launch_gemm) &&
                   all_pass;
    }
    if (!all_pass) {
        return EXIT_FAILURE;
    }

    constexpr int kWarmup = 5;
    constexpr int kIterations = 20;
    const std::vector<GemmShape> benchmark_shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };
    bool all_benchmarks_valid = true;
    for (const GemmShape shape : benchmark_shapes) {
        all_benchmarks_valid = run_gemm_benchmark_case(
                                   variant, shape, block,
                                   kWarmup, kIterations, launch_gemm) &&
                               all_benchmarks_valid;
    }
    return all_benchmarks_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
