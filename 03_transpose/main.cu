#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "cuda_check.h"

struct MatrixShape {
    int rows;
    int cols;
};

__global__ void transpose_naive_kernel(const float* input,
                                       float* output,
                                       int rows,
                                       int cols) {
    // 根据二维 block 和 thread 计算输入矩阵中的 row 和 col。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 检查边界后，将输入元素写入转置后矩阵的对应位置。
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

void transpose_cpu(const std::vector<float>& input,
                   std::vector<float>& expected,
                   int rows,
                   int cols) {
    // 按照 row-major 布局完成 CPU 转置参考实现。
    // 注意：输入形状是 [rows, cols]，输出形状是 [cols, rows]。
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            expected[j * rows + i] = input[i * cols + j];
        }
    }
}

float max_abs_error(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    // 检查元素数量并计算最大绝对误差；NaN/Inf 判定为失败。
    if (expected.size() != actual.size())
        return std::numeric_limits<float>::infinity();

    float max_err = 0.0F;
    for (std::size_t i = 0; i < expected.size(); i++) {
        float error = std::fabs(expected[i] - actual[i]);
        if (!std::isfinite(error)) {
            return std::numeric_limits<float>::infinity();
        }
        max_err = std::max(max_err, error);
    }
    return max_err;
}

bool run_correctness_case(int rows, int cols, dim3 block) {
    // 为一组矩阵执行 CPU Reference、GPU 转置和正确性验证。
    int n = rows * cols;
    std::size_t size = n * sizeof(float);

    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> expected(n);

    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
    }
    transpose_cpu(h_A, expected, rows, cols);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block>>>(d_A, d_B, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, size, cudaMemcpyDeviceToHost));

    float err = max_abs_error(expected, h_B);
    float tolerance = 1e-6F;
    bool pass = err <= tolerance;
    std::printf(
        "Input shape=(%d, %d), output shape=(%d, %d), block=(%u, %u, %u), grid=(%u, %u, %u), "
        "max_abs_error=%g, %s\n",
        rows, cols, cols, rows,
        block.x, block.y, block.z,
        grid.x, grid.y, grid.z,
        err,
        pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return pass;
}

float benchmark_transpose_naive(const float* d_input,
                                float* d_output,
                                int rows,
                                int cols,
                                dim3 block,
                                int warmup,
                                int iterations) {
    // 计算二维 grid。
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    // 执行 warm-up，检查 launch error，并在正式计时前同步。
    for (int i = 0; i < warmup; i++) {
        transpose_naive_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // 创建 start/stop CUDA Event。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // record(start) 后连续启动 iterations 次 kernel，再 record(stop)。
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        transpose_naive_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    // 计时循环内不调用 cudaDeviceSynchronize。
    // 检查 launch error、同步 stop Event、取得总毫秒数并销毁 Event。
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / iterations;
}

bool run_benchmark_case(int rows,
                        int cols,
                        dim3 block,
                        int warmup,
                        int iterations) {
    // 在计时区外准备输入和设备内存，再输出平均延迟与有效带宽。
    int n = rows * cols;
    std::size_t size = n * sizeof(float);

    std::vector<float> h_A(n);

    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
    }

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    float latency_ms = benchmark_transpose_naive(d_A, d_B, rows, cols, block, warmup, iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double bytes_per_kernel = rows * cols * 2.0 * sizeof(float);
    double bandwidth_GBps = (bytes_per_kernel) / (latency_ms * 1.0e6);

    std::printf(
        "shape=(%d, %d), block=(%u, %u, %u), grid=(%u, %u, %u), "
        "warmup=%d, iterations=%d, latency_ms=%.6fms, effective_bandwidth_GBps=%.2fGB/s\n",
        rows, cols,
        block.x, block.y, block.z,
        grid.x, grid.y, grid.z,
        warmup, iterations, latency_ms, bandwidth_GBps);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return std::isfinite(bandwidth_GBps) && bandwidth_GBps > 0 && std::isfinite(latency_ms) && latency_ms > 0;
}

int main() {
    const dim3 block(16, 16);
    const std::vector<MatrixShape> test_shapes = {
        {1, 1},
        {31, 33},
        {32, 33},
        {256, 256},
        {1000, 513},
    };

    bool all_pass = true;
    for (auto shape : test_shapes) {
        all_pass = all_pass & run_correctness_case(shape.rows, shape.cols, block);
    }
    if (!all_pass) {
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

    // 执行全部 benchmark，并汇总结果有效性。
    bool all_benchmarks_valid = true;
    for (const auto shape : benchmark_shapes) {
        all_benchmarks_valid =
            run_benchmark_case(shape.rows, shape.cols, block,
                               kWarmup, kIterations) &&
            all_benchmarks_valid;
    }
    return all_benchmarks_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
