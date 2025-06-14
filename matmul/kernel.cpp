#include "kernel.hpp"

// Baseline
void sgemm_v0(const float *const __restrict A, const float *const __restrict B,
              float *const __restrict C, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t k = 0; k < K; ++k) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

void sgemm_v1(const float *const __restrict A, const float *const __restrict B,
              float *const __restrict C, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float sum = .0f;
      for (uint32_t k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}
