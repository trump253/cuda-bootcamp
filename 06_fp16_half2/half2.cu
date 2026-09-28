#include "vector_add_harness.h"

// 你来完成：每个线程处理一对 half；若 N 为奇数，还要处理末尾一个元素。
__global__ void vector_add_half2_kernel(const __half* a, const __half* b,
                                        __half* c, std::size_t n) {
    // TODO：计算一维线程索引 i；合法 pair 满足 i < n / 2。
    // TODO：把输入与输出按 __half2* 访问，用 __hadd2 完成一对加法。
    // TODO：若 n 为奇数，让 i == n / 2 的线程用 __hadd 处理 c[n - 1]。
    // 注意：pair 路径不得访问奇数长度的最后一个元素之外的内存。
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* b2 = reinterpret_cast<const __half2*>(b);
    __half2* c2 = reinterpret_cast<__half2*>(c);

    std::size_t n_pairs = n / 2;

    if (idx < n_pairs) {
        c2[idx] = __hadd2(a2[idx], b2[idx]);
    }

    if ((n & 1) && idx == n_pairs) {
        c[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

int main(int argc, char** argv) {
    const auto launch = [](const __half* a, const __half* b, __half* c,
                           std::size_t n, dim3 grid, dim3 block) {
        vector_add_half2_kernel<<<grid, block>>>(a, b, c, n);
    };
    return run_vector_add_exercise<__half>("half2", 2, launch, argc, argv);
}
