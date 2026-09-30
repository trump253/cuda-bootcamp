cmake_minimum_required(VERSION 3.16)

project(cuda_bootcamp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CMake 3.16 能识别 CUDA C++14；CUDA 11.8 可以用这个设置正确编译当前代码。
# 如需提高到 CUDA C++17，请先升级 CMake。
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 当前两张 RTX 2080 Ti 均为计算能力 7.5。CMake 3.16 尚无
# CMAKE_CUDA_ARCHITECTURES；在目录级只给 CUDA 源文件传入目标架构。
# 所有当前与日后重新启用的 CUDA target 都会使用同一个配置。
# 换 GPU 时可在 CMake 配置阶段覆盖 CUDA_BOOTCAMP_ARCH。
set(CUDA_BOOTCAMP_ARCH "sm_75" CACHE STRING "本工程的 CUDA 编译目标架构")
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_BOOTCAMP_ARCH}>)

# Day 10 时间线练习：复用已验收的 Day 1 Vector Add。
add_executable(vector_add 01_vector_add/main.cu)
target_include_directories(vector_add PRIVATE common)

# Day 2 已完成；需要复查时取消以下两行注释。
# add_executable(matrix_add 02_matrix_add/main.cu)
# target_include_directories(matrix_add PRIVATE common)

# Naive Transpose V0 已完成；需要复查时取消以下两行注释。
# add_executable(transpose_naive 03_transpose/main.cu)
# target_include_directories(transpose_naive PRIVATE common)

function(add_transpose_tiled_target target_name tile_padding)
    add_executable(${target_name} 03_transpose/tiled.cu)
    target_include_directories(${target_name} PRIVATE common)
    target_compile_definitions(
        ${target_name} PRIVATE TRANSPOSE_TILE_PADDING=${tile_padding})
endfunction()

# Matrix Transpose 已完成；需要复查 Profile 时取消以下两行注释。
# add_transpose_tiled_target(transpose_tiled_padding0 0)
# add_transpose_tiled_target(transpose_tiled 1)

# Reduction V0 已完成；需要复查时取消以下两行注释。
# add_executable(reduction_v0 04_reduction/main.cu)
# target_include_directories(reduction_v0 PRIVATE common)

# Reduction V1 已完成；需要复查时取消以下两行注释。
# add_executable(reduction_v1 04_reduction/v1.cu)
# target_include_directories(reduction_v1 PRIVATE common)

# Reduction V2 已完成；需要复查 Profile 时取消以下两行注释。
# add_executable(reduction_v2 04_reduction/v2.cu)
# target_include_directories(reduction_v2 PRIVATE common)

# Day 10 时间线练习：复用 V3 的多轮依赖规约。
add_executable(reduction_v3 04_reduction/v3.cu)
target_include_directories(reduction_v3 PRIVATE common)

# Day 7 已完成；需要复查时取消以下目标的注释。
# add_executable(gemm_naive 05_gemm/naive.cu)
# target_include_directories(gemm_naive PRIVATE common 05_gemm)

# add_executable(gemm_tiled 05_gemm/tiled.cu)
# target_include_directories(gemm_tiled PRIVATE common 05_gemm)

# add_executable(gemm_cublas 05_gemm/cublas_compare.cu)
# target_include_directories(gemm_cublas PRIVATE common 05_gemm)
# target_link_libraries(gemm_cublas PRIVATE cublas)

# Day 8 已完成；需要复查时取消以下六行注释。
# add_executable(vector_add_fp32 06_fp16_half2/fp32.cu)
# target_include_directories(vector_add_fp32 PRIVATE common 06_fp16_half2)
# add_executable(vector_add_fp16 06_fp16_half2/fp16.cu)
# target_include_directories(vector_add_fp16 PRIVATE common 06_fp16_half2)
# add_executable(vector_add_half2 06_fp16_half2/half2.cu)
# target_include_directories(vector_add_half2 PRIVATE common 06_fp16_half2)

# Day 9 V0 已完成；需要复查时取消以下两行注释。
# add_executable(softmax_v0 07_softmax/v0.cu)
# target_include_directories(softmax_v0 PRIVATE common 07_softmax)

# Day 9 V1 已完成；需要复查历史对照时取消以下两行注释。
# add_executable(softmax_v1 07_softmax/v1.cu)
# target_include_directories(softmax_v1 PRIVATE common 07_softmax)

# Day 10 时间线练习：复用已验收的 Softmax V2。
add_executable(softmax_v2 07_softmax/v2.cu)
target_include_directories(softmax_v2 PRIVATE common 07_softmax)
# Day 9 FP32 V3 已完成；需要复查时取消以下两行注释。
# add_executable(softmax_v3 07_softmax/v3.cu)
# target_include_directories(softmax_v3 PRIVATE common 07_softmax)

# Day 9 FP16 已完成；同一源码可重新构建标量基线与 half2 练习版。
function(add_softmax_fp16_target target_name use_half2)
    add_executable(${target_name} 07_softmax/v3_fp16.cu)
    target_include_directories(${target_name} PRIVATE common 07_softmax)
    target_compile_definitions(
        ${target_name} PRIVATE SOFTMAX_FP16_USE_HALF2=${use_half2})
endfunction()

# add_softmax_fp16_target(softmax_fp16_scalar 0)
# add_softmax_fp16_target(softmax_fp16_half2 1)
