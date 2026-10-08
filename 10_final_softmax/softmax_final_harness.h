#pragma once

// 只复用已验证的主机输入生成、double CPU Reference、NaN 哨兵和误差检查。
// 该头文件不含 GPU kernel；不会引入 Day 9 的完整 GPU 实现。
#include "softmax_harness.h"

// TODO：独立补齐四个 Event 步骤；分配、暖机、循环和清理已提供。
template <typename Launch>
float measure_final_softmax_ms(const float* input, float* output,
                               int rows, int hidden, int warmup,
                               int iterations, Launch launch) {
    for (int i = 0; i < warmup; ++i) {
        launch(input, output, rows, hidden);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // TODO 1：在与 kernel 相同的 stream 上记录 start。
    for (int i = 0; i < iterations; ++i) {
        launch(input, output, rows, hidden);
    }
    // TODO 2：在相同 stream 上记录 stop。
    CUDA_CHECK(cudaGetLastError());
    // TODO 3：等待 stop 完成，确保 GPU 时间戳已记录。
    float total_ms = std::numeric_limits<float>::quiet_NaN();
    // TODO 4：取得 start/stop 的经过时间，写入 total_ms；单位是 ms。

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // 未完成计时时保持 NaN，测试支撑会拒绝输出虚假性能结果。
    return total_ms / iterations;
}

template <typename MakeConfig, typename Launch>
bool benchmark_final_softmax_case(const char* variant, int rows, int hidden,
                                  MakeConfig make_config, Launch launch) {
    constexpr int kWarmup = 10, kIterations = 100;
    const std::size_t count = static_cast<std::size_t>(rows) * hidden;
    const std::size_t bytes = count * sizeof(float);
    std::vector<float> input(count);
    initialize_softmax_input(input, rows, hidden);
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    const SoftmaxLaunchConfig config = make_config(rows, hidden);

    const float latency_ms = measure_final_softmax_ms(
        d_input, d_output, rows, hidden, kWarmup, kIterations, launch);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    if (!std::isfinite(latency_ms) || latency_ms <= 0.0F) {
        std::fprintf(stderr,
                     "计时未完成或结果无效：variant=%s, shape=(%d,%d)；"
                     "请检查四个 Event TODO，不记录此项为性能结果。\n",
                     variant, rows, hidden);
        return false;
    }
    std::printf("variant=%s, shape=(%d,%d), grid=(%u,%u,%u), "
                "block=(%u,%u,%u), warmup=%d, iterations=%d, "
                "latency=%.6f ms\n",
                variant, rows, hidden, config.grid.x, config.grid.y,
                config.grid.z, config.block.x, config.block.y, config.block.z,
                kWarmup, kIterations, latency_ms);
    return true;
}

template <typename MakeConfig, typename Launch>
int run_final_softmax(const char* variant, MakeConfig make_config,
                      Launch launch, int argc, char** argv) {
    const bool correctness_only =
        argc == 2 && std::strcmp(argv[1], "--correctness-only") == 0;
    const bool profile = argc == 2 && std::strcmp(argv[1], "--profile") == 0;
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("用法：%s [--correctness-only|--profile|--help]\n"
                    "默认：正确性通过后进行 Event Benchmark；"
                    "--profile：只启动一次 (128,4096) 目标 kernel 并验证输出。\n",
                    argv[0]);
        return 0;
    }
    if (argc != 1 && !correctness_only && !profile) {
        std::fprintf(stderr, "参数错误；运行 %s --help 查看用法。\n", argv[0]);
        return 2;
    }

    // 为兼容已验证的主机检查函数适配接口；启动配置仍由学习者定义。
    const auto check_launch = [launch](const float* input, float* output,
                                       int rows, int hidden,
                                       SoftmaxLaunchConfig /* config */) {
        launch(input, output, rows, hidden);
    };
    if (profile) {
        // 不运行暖机或其他尺寸，使 ncu --launch-count 1 选中正确用例。
        // 在计时区外复制结果做验证，空 kernel 同样必须 FAIL。
        std::printf("单次采集：variant=%s, shape=(128,4096)；"
                    "此模式不生成 Event Benchmark。\n", variant);
        return check_softmax_case(variant, 128, 4096,
                                  make_config, check_launch) ? 0 : 1;
    }

    bool all_passed = true;
    const int rows_cases[] = {1, 16, 128};
    const int hidden_cases[] = {128, 512, 1024, 4096};
    for (int rows : rows_cases) {
        for (int hidden : hidden_cases) {
            all_passed = check_softmax_case(variant, rows, hidden,
                                            make_config, check_launch)
                         && all_passed;
        }
    }
    const int boundaries[][2] = {
        {1, 1}, {1, 2}, {1, 31}, {5, 6}, {16, 33}, {128, 257},
        {129, 33}, {1, 255}, {16, 256}, {3, 513}, {16, 4097}};
    for (const auto& shape : boundaries) {
        all_passed = check_softmax_case(variant, shape[0], shape[1],
                                        make_config, check_launch)
                     && all_passed;
    }
    if (!all_passed) {
        std::fprintf(stderr, "最终验收正确性尚未通过，停止性能测试。\n");
        return 1;
    }
    if (correctness_only) return 0;

    const int benchmark_shapes[][2] =
        {{1, 128}, {16, 512}, {128, 1024}, {128, 4096}};
    for (const auto& shape : benchmark_shapes) {
        if (!benchmark_final_softmax_case(variant, shape[0], shape[1],
                                          make_config, launch)) return 1;
    }
    return 0;
}
