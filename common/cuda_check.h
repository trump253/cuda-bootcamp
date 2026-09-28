#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

// 检查 CUDA Runtime API 的返回值。发生错误时输出调用位置和错误详情，
// 然后立即终止程序。
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t error = (call);                                                   \
        if (error != cudaSuccess) {                                                         \
            std::cerr << "CUDA 错误位置：" << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::cerr << "  调用：" << #call << std::endl;                                  \
            std::cerr << "  错误码：" << static_cast<int>(error) << std::endl;               \
            std::cerr << "  错误名称：" << cudaGetErrorName(error) << std::endl;             \
            std::cerr << "  错误描述：" << cudaGetErrorString(error) << std::endl;           \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)
