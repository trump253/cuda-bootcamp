#pragma once

#include <cuda_fp16.h>

#include "softmax_harness.h"

// 输入先舍入为 FP16，再用实际存入 GPU 的数值生成 double CPU Reference。
// 这样比较的是 kernel 的计算，而不是输入量化本身的误差。
inline void initialize_softmax_fp16_input(std::vector<__half>& input,
                                          std::vector<float>& reference_input,
                                          int rows, int hidden) {
    initialize_softmax_input(reference_input, rows, hidden);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = __float2half_rn(reference_input[i]);
        reference_input[i] = __half2float(input[i]);
    }
}

inline SoftmaxLaunchConfig make_softmax_fp16_config(int rows) {
    return SoftmaxLaunchConfig{
        dim3(static_cast<unsigned int>(rows)), dim3(256)};
}

template <typename Launch>
bool check_softmax_fp16_case(const char* variant, int rows, int hidden,
                             Launch launch) {
    const std::size_t count = static_cast<std::size_t>(rows) * hidden;
    const std::size_t bytes = count * sizeof(__half);
    std::vector<__half> input(count), actual(count);
    std::vector<float> reference_input(count);
    std::vector<double> expected(count);
    initialize_softmax_fp16_input(input, reference_input, rows, hidden);
    cpu_softmax_reference(reference_input, expected, rows, hidden);

    __half *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    // 半精度 NaN 哨兵：漏写不能碰巧通过正确性检查。
    CUDA_CHECK(cudaMemset(d_output, 0xFF, bytes));

    const SoftmaxLaunchConfig config = make_softmax_fp16_config(rows);
    launch(d_input, d_output, rows, hidden, config);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    // FP16 输出会舍入；阈值与 FP32 练习分开，不用于掩盖漏写或 NaN。
    constexpr double kAbsoluteTolerance = 3.0e-4;
    constexpr double kRelativeTolerance = 1.0e-3;
    constexpr double kRowSumTolerance = 1.0e-3;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double max_row_sum_error = 0.0;
    bool passed = true;
    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < hidden; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * hidden + col;
            const double value = static_cast<double>(__half2float(actual[index]));
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

template <typename Launch>
void benchmark_softmax_fp16_case(const char* variant, int rows, int hidden,
                                 Launch launch) {
    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;
    const std::size_t count = static_cast<std::size_t>(rows) * hidden;
    const std::size_t bytes = count * sizeof(__half);
    std::vector<__half> input(count);
    std::vector<float> reference_input(count);
    initialize_softmax_fp16_input(input, reference_input, rows, hidden);

    __half *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    const SoftmaxLaunchConfig config = make_softmax_fp16_config(rows);
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

template <typename Launch>
int run_softmax_fp16_exercise(const char* variant, Launch launch,
                             int argc, char** argv) {
    const bool correctness_only =
        argc == 2 && std::strcmp(argv[1], "--correctness-only") == 0;
    const bool profile = argc == 2 && std::strcmp(argv[1], "--profile") == 0;
    if (argc != 1 && !correctness_only && !profile) {
        std::fprintf(stderr, "用法：%s [--correctness-only|--profile]\n", argv[0]);
        return 2;
    }
    if (profile) {
        // 单次 kernel 启动，保留与 FP32 相同的 Profile 形状。
        constexpr int rows = 128, hidden = 4096;
        const std::size_t count = static_cast<std::size_t>(rows) * hidden;
        const std::size_t bytes = count * sizeof(__half);
        std::vector<__half> input(count);
        std::vector<float> reference_input(count);
        initialize_softmax_fp16_input(input, reference_input, rows, hidden);
        __half *d_input = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
        const SoftmaxLaunchConfig config = make_softmax_fp16_config(rows);
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
            all_passed = check_softmax_fp16_case(
                variant, rows, hidden, launch) && all_passed;
        }
    }
    const int boundary_cases[][2] =
        {{1, 1}, {1, 2}, {1, 31}, {5, 6},
         {16, 33}, {128, 257}, {129, 33}};
    for (const auto& shape : boundary_cases) {
        all_passed = check_softmax_fp16_case(
            variant, shape[0], shape[1], launch) && all_passed;
    }
    if (!all_passed) return 1;
    if (!correctness_only) {
        const int benchmark_cases[][2] =
            {{1, 128}, {16, 512}, {128, 1024}, {128, 4096}};
        for (const auto& shape : benchmark_cases) {
            benchmark_softmax_fp16_case(
                variant, shape[0], shape[1], launch);
        }
    }
    return 0;
}
