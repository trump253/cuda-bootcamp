#include <cuda_runtime.h>

#include "gemm_harness.h"

constexpr int kTileSize = 16;

__global__ void gemm_tiled_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int m,
                                  int n,
                                  int k) {
    __shared__ float tile_a[kTileSize][kTileSize];
    __shared__ float tile_b[kTileSize][kTileSize];

    // 每个线程负责一个输出元素，并保留线程私有累加值。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float accumulator = 0.0f;

    // 分阶段遍历 K 维，越界元素写 0；所有线程都要到达 block barrier。
    int nums_tiles = (k + kTileSize - 1) / kTileSize;
    for (int tile = 0; tile < nums_tiles; tile++) {
        int k0 = tile * kTileSize;
        int a_col = k0 + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;

        int b_row = k0 + threadIdx.y;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < kTileSize; i++) {
            accumulator += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }
    // 全部阶段结束后，只写回有效的 C 元素。
    if (row < m && col < n) {
        c[row * n + col] = accumulator;
    }
}

void launch_gemm_tiled(const float* d_a,
                       const float* d_b,
                       float* d_c,
                       GemmShape shape,
                       dim3 block) {
    const dim3 grid(
        (shape.n + static_cast<int>(block.x) - 1) / block.x,
        (shape.m + static_cast<int>(block.y) - 1) / block.y);
    gemm_tiled_kernel<<<grid, block>>>(
        d_a, d_b, d_c, shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    const dim3 block(kTileSize, kTileSize);
    return run_gemm_program(
        argc, argv, "v2_shared_tiled", block, launch_gemm_tiled);
}
