#include <cuda_runtime.h>

#include "cuda_check.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

constexpr int kBlockSize = 256;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFU;
static_assert(kBlockSize % kWarpSize == 0,
              "block size 必须是 warp size 的整数倍");

__device__ float warp_reduce_sum(float value, unsigned int mask) {
    // delta 依次减半，每轮把当前 value 与更高 lane 的 value 相加。
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(mask, value, delta);
    }
    return value;
}

__global__ void reduce_sum_v3_kernel(const float* input,
                                     float* block_sums,
                                     std::size_t n) {
    __shared__ float warp_sums[kWarpsPerBlock];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % kWarpSize;
    const unsigned int warp_id = tid / kWarpSize;
    const std::size_t global_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;

    // 尾部线程贡献加法单位元 0；所有线程仍参加后续 warp shuffle。
    float value = global_idx < n ? input[global_idx] : 0.0F;

    // 当前 block 固定为 256 个线程，所有 32 个 lane 都存在，因此使用 full mask。
    float warp_sum = warp_reduce_sum(value, kFullWarpMask);

    // 每个 warp 的 lane 0 把 warp partial sum 写入 shared memory。
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }

    // 所有 warp partial sum 写入完成后，第一 warp 才能读取。
    __syncthreads();

    // 只由 warp 0 完成 block 的最终归约。
    // lane_id < kWarpsPerBlock 时读取对应 warp sum，其余 lane 读取 0。
    // warp 0 的全部 lane 再调用一次 warp_reduce_sum，最终由 lane 0 写回 block_sums。
    if (warp_id == 0) {
        value = lane_id < kWarpsPerBlock ? warp_sums[lane_id] : 0.0F;
        float sum = warp_reduce_sum(value, kFullWarpMask);
        if (lane_id == 0) {
            block_sums[blockIdx.x] = sum;
        }
    }
}

double reduce_sum_cpu(const std::vector<float>& input) {
    double sum = 0.0;
    for (const float value : input) {
        sum += static_cast<double>(value);
    }
    return sum;
}

void initialize_input(std::vector<float>& input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = 1.0F + static_cast<float>(i % 7) * 0.125F;
    }
}

std::size_t divide_round_up(std::size_t value, std::size_t divisor) {
    return (value + divisor - 1) / divisor;
}

int count_reduction_passes(std::size_t n, int block_size) {
    int passes = 0;
    while (n > 1) {
        n = divide_round_up(n, static_cast<std::size_t>(block_size));
        ++passes;
    }
    return passes;
}

const float* launch_reduce_sum_v3(const float* d_input,
                                  float* d_partial_a,
                                  float* d_partial_b,
                                  std::size_t n,
                                  int block_size) {
    const float* current_input = d_input;
    float* current_output = d_partial_a;
    std::size_t current_n = n;
    bool write_to_a = true;
    // 每一轮将 current_n 个输入压缩为 ceil(current_n / block_size) 个 partial sum，
    // 并在两个临时缓冲区之间 ping-pong，直到设备端只剩一个标量。
    while (current_n > 1) {
        const std::size_t blocks = divide_round_up(
            current_n, static_cast<std::size_t>(block_size));
        reduce_sum_v3_kernel<<<static_cast<unsigned int>(blocks), block_size>>>(
            current_input, current_output, current_n);
        CUDA_CHECK(cudaGetLastError());

        current_input = current_output;
        current_n = blocks;
        write_to_a = !write_to_a;
        current_output = write_to_a ? d_partial_a : d_partial_b;
    }
    return current_input;
}

bool run_correctness_case(std::size_t n, int block_size) {
    if (n == 0 || block_size <= 0) {
        return false;
    }

    const std::size_t input_bytes = n * sizeof(float);
    const std::size_t partial_capacity = divide_round_up(
        n, static_cast<std::size_t>(block_size));
    const std::size_t partial_bytes = partial_capacity * sizeof(float);

    std::vector<float> input(n);
    initialize_input(input);
    const double expected = reduce_sum_cpu(input);

    float* d_input = nullptr;
    float* d_partial_a = nullptr;
    float* d_partial_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_a, partial_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_b, partial_bytes));
    CUDA_CHECK(cudaMemcpy(
        d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partial_a, 0, partial_bytes));
    CUDA_CHECK(cudaMemset(d_partial_b, 0, partial_bytes));

    const float* d_result = launch_reduce_sum_v3(
        d_input, d_partial_a, d_partial_b, n, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    float actual = std::numeric_limits<float>::quiet_NaN();
    CUDA_CHECK(cudaMemcpy(
        &actual, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    const double abs_error =
        std::fabs(expected - static_cast<double>(actual));
    constexpr double kTolerance = 1e-5;
    const bool passed = std::isfinite(actual) && abs_error <= kTolerance;
    const std::size_t first_grid = divide_round_up(
        n, static_cast<std::size_t>(block_size));
    std::printf(
        "variant=v3_warp_shuffle, N=%zu, first_grid=%zu, block=%d, passes=%d, "
        "expected=%.6f, actual=%.6f, abs_error=%.6g, %s\n",
        n, first_grid, block_size, count_reduction_passes(n, block_size),
        expected, static_cast<double>(actual), abs_error,
        passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial_a));
    CUDA_CHECK(cudaFree(d_partial_b));
    return passed;
}

float benchmark_reduce_sum_v3(const float* d_input,
                              float* d_partial_a,
                              float* d_partial_b,
                              std::size_t n,
                              int block_size,
                              int warmup,
                              int iterations) {
    if (n == 0 || block_size <= 0 || warmup < 0 || iterations <= 0) {
        return std::numeric_limits<float>::infinity();
    }

    for (int i = 0; i < warmup; ++i) {
        launch_reduce_sum_v3(
            d_input, d_partial_a, d_partial_b, n, block_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_reduce_sum_v3(
            d_input, d_partial_a, d_partial_b, n, block_size);
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

bool run_benchmark_case(std::size_t n,
                        int block_size,
                        int warmup,
                        int iterations) {
    const std::size_t input_bytes = n * sizeof(float);
    const std::size_t partial_capacity = divide_round_up(
        n, static_cast<std::size_t>(block_size));
    const std::size_t partial_bytes = partial_capacity * sizeof(float);

    std::vector<float> input(n);
    initialize_input(input);

    float* d_input = nullptr;
    float* d_partial_a = nullptr;
    float* d_partial_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_a, partial_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_b, partial_bytes));
    CUDA_CHECK(cudaMemcpy(
        d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partial_a, 0, partial_bytes));
    CUDA_CHECK(cudaMemset(d_partial_b, 0, partial_bytes));

    const float latency_ms = benchmark_reduce_sum_v3(
        d_input, d_partial_a, d_partial_b, n, block_size,
        warmup, iterations);
    // 为便于版本间比较，有效带宽统一只计算原始输入的 N × sizeof(float)。
    const double effective_bandwidth_gbps =
        static_cast<double>(input_bytes) /
        (static_cast<double>(latency_ms) * 1.0e6);
    const std::size_t first_grid = divide_round_up(
        n, static_cast<std::size_t>(block_size));
    std::printf(
        "variant=v3_warp_shuffle, N=%zu, first_grid=%zu, block=%d, passes=%d, "
        "warmup=%d, iterations=%d, latency=%.6f ms, "
        "effective_bandwidth=%.2f GB/s\n",
        n, first_grid, block_size, count_reduction_passes(n, block_size),
        warmup, iterations, latency_ms, effective_bandwidth_gbps);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial_a));
    CUDA_CHECK(cudaFree(d_partial_b));
    return std::isfinite(latency_ms) && latency_ms > 0.0F &&
           std::isfinite(effective_bandwidth_gbps) &&
           effective_bandwidth_gbps > 0.0;
}

bool run_profile_case(std::size_t n, int block_size) {
    const std::size_t input_bytes = n * sizeof(float);
    const std::size_t blocks = divide_round_up(
        n, static_cast<std::size_t>(block_size));
    const std::size_t output_bytes = blocks * sizeof(float);
    float* d_input = nullptr;
    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_sums, output_bytes));
    CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));

    // Profile 模式只采集第一阶段的一次 kernel，避免混入后续小规模 pass。
    reduce_sum_v3_kernel<<<static_cast<unsigned int>(blocks), block_size>>>(
        d_input, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf(
        "profile variant=v3_warp_shuffle, N=%zu, grid=%zu, block=%d\n",
        n, blocks, block_size);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_block_sums));
    return true;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--profile") == 0) {
        return run_profile_case(1U << 24, kBlockSize)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (argc == 2 && std::strcmp(argv[1], "--timeline") == 0) {
        // Day 10：只运行一次多轮规约，保留 H2D、各轮 kernel 与 D2H。
        // 目标尺寸依次产生 4097、17、1 个 partial sum。
        return run_correctness_case((1U << 20) + 3, kBlockSize)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (argc != 1) {
        std::fprintf(stderr, "用法：%s [--profile|--timeline]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::vector<std::size_t> correctness_sizes = {
        1,
        31,
        32,
        33,
        255,
        256,
        257,
        1000,
        (1U << 20) + 3,
    };

    bool all_correct = true;
    for (const std::size_t n : correctness_sizes) {
        all_correct = run_correctness_case(n, kBlockSize) && all_correct;
    }
    if (!all_correct) {
        return EXIT_FAILURE;
    }

    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;
    const std::vector<std::size_t> benchmark_sizes = {
        1U << 10,
        1U << 14,
        1U << 18,
        1U << 20,
        1U << 24,
    };

    bool all_benchmarks_valid = true;
    for (const std::size_t n : benchmark_sizes) {
        all_benchmarks_valid =
            run_benchmark_case(n, kBlockSize, kWarmup, kIterations) &&
            all_benchmarks_valid;
    }
    return all_benchmarks_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
