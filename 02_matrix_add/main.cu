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

__global__ void matrix_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int rows,
                                  int cols) {
    // 根据二维 block 和 thread 计算当前线程负责的行、列。
    // 行、列均在有效范围内时，完成对应元素的加法。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

void matrix_add_cpu(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c,
                    int rows,
                    int cols) {
    // 按照 row-major 布局计算 CPU 参考结果。
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            c[idx] = a[idx] + b[idx];
        }
    }
}

float max_abs_error(const std::vector<float>& expected,
                    const std::vector<float>& actual) {
    // 检查两个结果的元素数量，并计算最大绝对误差。
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
    // 为一组矩阵执行 CPU Reference、GPU 计算和正确性验证。
    int n = rows * cols;
    std::size_t size = n * sizeof(float);

    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_C(n);
    std::vector<float> expected(n);

    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    matrix_add_cpu(h_A, h_B, expected, rows, cols);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matrix_add_kernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    float err = max_abs_error(expected, h_C);
    float tolerance = 1e-6F;
    bool pass = err <= tolerance;
    std::printf(
        "shape=(%d, %d), block=(%u, %u, %u), grid=(%u, %u, %u), "
        "max_abs_error=%g, %s\n",
        rows, cols,
        block.x, block.y, block.z,
        grid.x, grid.y, grid.z,
        err,
        pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return pass;
}

float benchmark_matrix_add(const float* d_a,
                           const float* d_b,
                           float* d_c,
                           int rows,
                           int cols,
                           dim3 block,
                           int warmup,
                           int iterations) {
    // 计算二维 grid。
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    // 执行 warm-up，并在正式计时前完成错误检查与同步。
    for (int i = 0; i < warmup; i++) {
        matrix_add_kernel<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 创建 start 和 stop 两个 CUDA Event。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // 在同一个 CUDA stream 上按以下顺序提交：
    //       record(start) -> iterations 次 kernel -> record(stop)。
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        matrix_add_kernel<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    // 计时循环不包含同步、内存分配或数据复制。
    // 检查正式计时阶段的 kernel launch，并等待 stop Event 完成。
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
    // 在计时区外准备数据和设备内存，再输出平均延迟与有效带宽。
    int n = rows * cols;
    std::size_t size = n * sizeof(float);

    std::vector<float> h_A(n);
    std::vector<float> h_B(n);

    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    float latency_ms = benchmark_matrix_add(d_A, d_B, d_C, rows, cols, block, warmup, iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double bytes_per_kernel = rows * cols * 3.0 * sizeof(float);
    double bandwidth_GBps = (bytes_per_kernel) / (latency_ms * 1.0e6);

    std::printf(
        "shape=(%d, %d), block=(%u, %u, %u), grid=(%u, %u, %u), "
        "warmup=%d, iterations=%d, latency_ms=%.6f, effective_bandwidth_GBps=%.2f\n",
        rows, cols,
        block.x, block.y, block.z,
        grid.x, grid.y, grid.z,
        warmup, iterations, latency_ms, bandwidth_GBps);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return std::isfinite(bandwidth_GBps) && bandwidth_GBps > 0 && std::isfinite(latency_ms) && latency_ms > 0;
}

int main() {
    const dim3 block(16, 16);
    const std::vector<MatrixShape> test_shapes = {
        {1, 1},
        {31, 33},
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

    // 依次执行全部 benchmark，并汇总结果有效性。
    bool all_benchmarks_valid = true;
    for (const auto shape : benchmark_shapes) {
        all_benchmarks_valid =
            run_benchmark_case(shape.rows, shape.cols, block,
                               kWarmup, kIterations) &&
            all_benchmarks_valid;
    }
    return all_benchmarks_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
