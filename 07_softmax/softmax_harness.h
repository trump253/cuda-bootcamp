#pragma once

#include <cuda_runtime.h>

#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

// Day 9 各版本共用的主机端测试框架；GPU Softmax 算法仍由学习者实现。
struct SoftmaxLaunchConfig {
    dim3 grid;
    dim3 block;
};

inline void initialize_softmax_input(std::vector<float>& input,
                                     int rows, int hidden) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < hidden; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * hidden + col;
            // 同时覆盖大正数、大负数和完全相同的一行，暴露不稳定的 exp(x)。
            const float offset = row % 3 == 0 ? 100.0F
                                 : row % 3 == 1 ? -100.0F : 0.0F;
            const float value = static_cast<float>((col * 17 + row * 13) % 113 - 56)
                                * 0.125F;
            input[index] = row % 11 == 7 ? offset : offset + value;
        }
    }
}

inline void cpu_softmax_reference(const std::vector<float>& input,
                                  std::vector<double>& expected,
                                  int rows, int hidden) {
    for (int row = 0; row < rows; ++row) {
        const std::size_t start = static_cast<std::size_t>(row) * hidden;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < hidden; ++col) {
            maximum = std::max(maximum, static_cast<double>(input[start + col]));
        }
        double sum = 0.0;
        for (int col = 0; col < hidden; ++col) {
            const double value = std::exp(static_cast<double>(input[start + col])
                                          - maximum);
            expected[start + col] = value;
            sum += value;
        }
        for (int col = 0; col < hidden; ++col) {
            expected[start + col] /= sum;
        }
    }
}

template <typename MakeConfig, typename Launch>
bool check_softmax_case(const char* variant, int rows, int hidden,
                        MakeConfig make_config, Launch launch) {
    const std::size_t count = static_cast<std::size_t>(rows) * hidden;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count), actual(count);
    std::vector<double> expected(count);
    initialize_softmax_input(input, rows, hidden);
    cpu_softmax_reference(input, expected, rows, hidden);

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    // 未写入的位置保留 NaN；不能用默认的 0 误判为正确结果。
    CUDA_CHECK(cudaMemset(d_output, 0xFF, bytes));

    const SoftmaxLaunchConfig config = make_config(rows, hidden);
    launch(d_input, d_output, rows, hidden, config);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    constexpr double kAbsoluteTolerance = 2.0e-5;
    constexpr double kRelativeTolerance = 1.0e-4;
    constexpr double kRowSumTolerance = 2.0e-4;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double max_row_sum_error = 0.0;
    bool passed = true;
    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < hidden; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * hidden + col;
            const double value = static_cast<double>(actual[index]);
            const double reference = expected[index];
            if (!std::isfinite(value)) {
                passed = false;
                max_absolute_error = std::numeric_limits<double>::infinity();
                max_relative_error = std::numeric_limits<double>::infinity();
                continue;
            }
            row_sum += value;
            const double absolute_error = std::fabs(value - reference);
            const double relative_error =
                absolute_error / std::max(std::fabs(reference), 1.0e-6);
            max_absolute_error = std::max(max_absolute_error, absolute_error);
            max_relative_error = std::max(max_relative_error, relative_error);
            if (value < 0.0 || value > 1.0 + kAbsoluteTolerance ||
                absolute_error > kAbsoluteTolerance
                                  + kRelativeTolerance * std::fabs(reference)) {
                passed = false;
            }
        }
        const double sum_error = std::fabs(row_sum - 1.0);
        max_row_sum_error = std::max(max_row_sum_error, sum_error);
        if (sum_error > kRowSumTolerance) passed = false;
    }
    std::printf("variant=%s, shape=(%d,%d), grid=%u, block=%u, "
                "max_abs_error=%.6g, max_rel_error=%.6g, "
                "max_row_sum_error=%.6g, %s\n",
                variant, rows, hidden, config.grid.x, config.block.x,
                max_absolute_error, max_relative_error, max_row_sum_error,
                passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return passed;
}

template <typename MakeConfig, typename Launch>
void benchmark_softmax_case(const char* variant, int rows, int hidden,
                            MakeConfig make_config, Launch launch) {
    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;
    const std::size_t count = static_cast<std::size_t>(rows) * hidden;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count);
    initialize_softmax_input(input, rows, hidden);
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    const SoftmaxLaunchConfig config = make_config(rows, hidden);

    for (int i = 0; i < kWarmup; ++i) {
        launch(d_input, d_output, rows, hidden, config);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        launch(d_input, d_output, rows, hidden, config);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    std::printf("variant=%s, shape=(%d,%d), grid=%u, block=%u, "
                "warmup=%d, iterations=%d, latency=%.6f ms\n",
                variant, rows, hidden, config.grid.x, config.block.x,
                kWarmup, kIterations, total_ms / kIterations);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

template <typename MakeConfig, typename Launch>
int run_softmax_exercise(const char* variant, MakeConfig make_config,
                         Launch launch, int argc, char** argv) {
    const bool correctness_only =
        argc == 2 && std::strcmp(argv[1], "--correctness-only") == 0;
    const bool profile = argc == 2 && std::strcmp(argv[1], "--profile") == 0;
    if (argc != 1 && !correctness_only && !profile) {
        std::fprintf(stderr, "用法：%s [--correctness-only|--profile]\n", argv[0]);
        return 2;
    }
    if (profile) {
        // 单次目标 kernel 启动，避免采集工具误选 warm-up 或测试用例。
        constexpr int rows = 128, hidden = 4096;
        const std::size_t bytes = static_cast<std::size_t>(rows) * hidden
                                  * sizeof(float);
        std::vector<float> input(static_cast<std::size_t>(rows) * hidden);
        initialize_softmax_input(input, rows, hidden);
        float *d_input = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
        const SoftmaxLaunchConfig config = make_config(rows, hidden);
        launch(d_input, d_output, rows, hidden, config);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("profile variant=%s, shape=(%d,%d), grid=%u, block=%u\n",
                    variant, rows, hidden, config.grid.x, config.block.x);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return 0;
    }

    const int rows_cases[] = {1, 16, 128};
    const int hidden_cases[] = {128, 512, 1024, 4096};
    bool all_passed = true;
    for (int rows : rows_cases) {
        for (int hidden : hidden_cases) {
            all_passed = check_softmax_case(variant, rows, hidden,
                                             make_config, launch) && all_passed;
        }
    }
    const int boundary_cases[][2] =
        {{1, 1}, {1, 2}, {1, 31}, {5, 6},
         {16, 33}, {128, 257}, {129, 33}};
    for (const auto& shape : boundary_cases) {
        all_passed = check_softmax_case(variant, shape[0], shape[1],
                                         make_config, launch) && all_passed;
    }
    if (!all_passed) return 1;
    if (!correctness_only) {
        const int benchmark_cases[][2] =
            {{1, 128}, {16, 512}, {128, 1024}, {128, 4096}};
        for (const auto& shape : benchmark_cases) {
            benchmark_softmax_case(variant, shape[0], shape[1],
                                   make_config, launch);
        }
    }
    return 0;
}
