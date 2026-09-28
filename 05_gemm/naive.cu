#include <cuda_runtime.h>

#include "gemm_harness.h"

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void gemm_naive_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int m,
                                  int n,
                                  int k) {
    // 根据二维 block/thread 计算当前线程负责的 C[row, col]。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 进行行列边界检查，沿 K 维计算点积并写入 C。
    // 矩阵均为 row-major：A=(M,K)、B=(K,N)、C=(M,N)。
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void launch_gemm_naive(const float* d_a,
                       const float* d_b,
                       float* d_c,
                       GemmShape shape,
                       dim3 block) {
    const dim3 grid(
        (shape.n + static_cast<int>(block.x) - 1) / block.x,
        (shape.m + static_cast<int>(block.y) - 1) / block.y);
    gemm_naive_kernel<<<grid, block>>>(
        d_a, d_b, d_c, shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    const dim3 block(kBlockX, kBlockY);
    return run_gemm_program(
        argc, argv, "v1_naive", block, launch_gemm_naive);
}
