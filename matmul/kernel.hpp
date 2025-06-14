#ifndef MATMUL_KERNEL_HPP
#define MATMUL_KERNEL_HPP

#include <cstdint>

void sgemm_v0(const float *const __restrict A, const float *const __restrict B,
              float *const __restrict C, uint32_t M, uint32_t N, uint32_t K);

void sgemm_v1(const float *const __restrict A, const float *const __restrict B,
              float *const __restrict C, uint32_t M, uint32_t N, uint32_t K);

#endif