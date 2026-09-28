#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemm_harness.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        const cublasStatus_t status = (call);                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS 调用失败：%s，状态码=%d\n", #call,      \
                         static_cast<int>(status));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

void launch_cublas_row_major_sgemm(cublasHandle_t handle,
                                   const float* d_a,
                                   const float* d_b,
                                   float* d_c,
                                   GemmShape shape) {
    const float alpha = 1.0F;
    const float beta = 0.0F;

    // cuBLAS 将 row-major B(K,N) 解释为 column-major B^T(N,K)，
    // 将 row-major A(M,K) 解释为 column-major A^T(K,M)。
    // 因此计算 C^T(N,M)=B^T(N,K)×A^T(K,M)，结果内存正好是 row-major C(M,N)。
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        shape.n, shape.m, shape.k,
        &alpha, d_b, shape.n,
        d_a, shape.k,
        &beta, d_c, shape.n));
}

int main(int argc, char** argv) {
    if (argc != 1) {
        std::fprintf(stderr, "用法：%s\n", argv[0]);
        return EXIT_FAILURE;
    }

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, 0));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // 复用 Day 7 的 CPU Reference、输入、误差判据和 CUDA Event 测量。
    // cuBLAS 自行选择 launch 配置，因此不打印手写 kernel 的 block/grid。
    const dim3 unused_block(1, 1);
    const auto launch = [handle](const float* d_a,
                                 const float* d_b,
                                 float* d_c,
                                 GemmShape shape,
                                 dim3) {
        launch_cublas_row_major_sgemm(handle, d_a, d_b, d_c, shape);
    };

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
                       "cublas_sgemm", shape, unused_block, launch,
                       false) &&
                   all_pass;
    }

    bool benchmark_valid = false;
    if (all_pass) {
        const GemmShape benchmark_shape{1024, 1024, 1024};
        constexpr int kWarmup = 5;
        constexpr int kIterations = 20;
        benchmark_valid = run_gemm_benchmark_case(
            "cublas_sgemm", benchmark_shape, unused_block,
            kWarmup, kIterations, launch, false);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return all_pass && benchmark_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
