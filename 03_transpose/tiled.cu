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

constexpr int kTileDim = 16;
#ifndef TRANSPOSE_TILE_PADDING
#define TRANSPOSE_TILE_PADDING 1
#endif
// 由 CMake 为对照实验分别设置为 0 和 1，kernel 的其余部分保持完全一致。
constexpr int kTilePadding = TRANSPOSE_TILE_PADDING;
static_assert(kTilePadding == 0 || kTilePadding == 1,
              "当前实验只比较 padding=0 和 padding=1");

struct MatrixShape {
    int rows;
    int cols;
};

__global__ void transpose_tiled_kernel(const float* input,
                                       float* output,
                                       int rows,
                                       int cols) {
    __shared__ float tile[kTileDim][kTileDim + kTilePadding];

    // 计算本线程从 input 读取时使用的全局 row、col。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界内的线程以合并方式将 input 写入 tile。
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    // 整个 block 都必须到达同步点，不能把 barrier 放入部分线程才进入的分支。
    __syncthreads();
    // 交换 block 坐标，并让 threadIdx.x 对应连续的 output col。
    col = blockIdx.y * blockDim.y + threadIdx.x;
    row = blockIdx.x * blockDim.x + threadIdx.y;
    // 从 tile 的转置位置读取，并以合并方式写入 output。
    if (row < cols && col < rows) {
        output[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_cpu(const std::vector<float>& input,
                   std::vector<float>& expected,
                   int rows,
                   int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            expected[static_cast<std::size_t>(col) * rows + row] =
                input[static_cast<std::size_t>(row) * cols + col];
        }
    }
}

float max_abs_error(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        return std::numeric_limits<float>::infinity();
    }

    float max_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::fabs(expected[i] - actual[i]);
        if (!std::isfinite(error)) {
            return std::numeric_limits<float>::infinity();
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

bool run_tiled_correctness_case(int rows, int cols, dim3 block) {
    constexpr float kTolerance = 1e-6F;
    if (rows <= 0 || cols <= 0 ||
        block.x != kTileDim || block.y != kTileDim) {
        return false;
    }

    const std::size_t element_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t bytes = element_count * sizeof(float);

    std::vector<float> input(element_count);
    std::vector<float> actual(element_count);
    std::vector<float> expected(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        input[i] = static_cast<float>(i);
    }
    transpose_cpu(input, expected, rows, cols);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0xFF, bytes));

    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    const float error = max_abs_error(expected, actual);
    const bool passed = error <= kTolerance;
    std::printf(
        "variant=tiled_padding_%d, input=(%d,%d), output=(%d,%d), "
        "block=(%u,%u), grid=(%u,%u), max_abs_error=%g, %s\n",
        kTilePadding, rows, cols, cols, rows,
        block.x, block.y, grid.x, grid.y,
        error, passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return passed;
}

float benchmark_transpose_tiled(const float* d_input,
                                float* d_output,
                                int rows,
                                int cols,
                                dim3 block,
                                int warmup,
                                int iterations) {
    if (warmup < 0 || iterations <= 0) {
        return std::numeric_limits<float>::infinity();
    }

    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    for (int i = 0; i < warmup; ++i) {
        transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
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

bool run_tiled_benchmark_case(int rows,
                              int cols,
                              dim3 block,
                              int warmup,
                              int iterations) {
    const std::size_t element_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t bytes = element_count * sizeof(float);
    std::vector<float> input(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        input[i] = static_cast<float>(i);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    const float latency_ms =
        benchmark_transpose_tiled(d_input, d_output, rows, cols,
                                  block, warmup, iterations);
    const double bytes_per_kernel = static_cast<double>(element_count) *
                                    2.0 * sizeof(float);
    const double bandwidth_gbps = bytes_per_kernel / (latency_ms * 1.0e6);
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);

    std::printf(
        "variant=tiled_padding_%d, shape=(%d,%d), block=(%u,%u), "
        "grid=(%u,%u), warmup=%d, iterations=%d, latency_ms=%.6fms, "
        "effective_bandwidth_GBps=%.2fGB/s\n",
        kTilePadding, rows, cols, block.x, block.y, grid.x, grid.y,
        warmup, iterations, latency_ms, bandwidth_gbps);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return std::isfinite(latency_ms) && latency_ms > 0.0F &&
           std::isfinite(bandwidth_gbps) && bandwidth_gbps > 0.0;
}

bool run_tiled_profile_case(int rows, int cols, dim3 block) {
    const std::size_t element_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t bytes = element_count * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemset(d_input, 0, bytes));

    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    // Profile 模式只启动一次目标 kernel，避免采集正确性与 Benchmark 的重复 launch。
    transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf(
        "profile variant=tiled_padding_%d, shape=(%d,%d), "
        "block=(%u,%u), grid=(%u,%u)\n",
        kTilePadding, rows, cols, block.x, block.y, grid.x, grid.y);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return true;
}

int main(int argc, char** argv) {
    const dim3 block(kTileDim, kTileDim);
    if (argc == 2 && std::strcmp(argv[1], "--profile") == 0) {
        return run_tiled_profile_case(4096, 4096, block)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (argc != 1) {
        std::fprintf(stderr, "用法：%s [--profile]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::vector<MatrixShape> correctness_shapes = {
        {1, 1},
        {15, 17},
        {31, 33},
        {32, 33},
        {256, 256},
        {1000, 513},
    };

    bool all_correct = true;
    for (const auto shape : correctness_shapes) {
        all_correct =
            run_tiled_correctness_case(shape.rows, shape.cols, block) &&
            all_correct;
    }
    if (!all_correct) {
        return EXIT_FAILURE;
    }

    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;
    const std::vector<MatrixShape> benchmark_shapes = {
        {256, 256},
        {1000, 513},
        {1024, 1024},
        {4096, 4096},
    };

    bool all_benchmarks_valid = true;
    for (const auto shape : benchmark_shapes) {
        all_benchmarks_valid =
            run_tiled_benchmark_case(shape.rows, shape.cols, block,
                                     kWarmup, kIterations) &&
            all_benchmarks_valid;
    }
    return all_benchmarks_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
