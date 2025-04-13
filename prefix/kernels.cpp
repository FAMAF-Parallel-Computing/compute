#include "kernels.hpp"

void prefix_scan(uint64_t n, const float *const __restrict src,
                 float *const __restrict dst, const double init) {
  auto acc = init;
  for (uint64_t i = 0; i < n; ++i) {
    acc += src[i];
    dst[i] = acc;
  }
}

void prefix_scan_omp_simd(uint64_t n, const float *const __restrict src,
                          float *const __restrict dst, const double init) {
  auto acc = init;
#pragma omp simd reduction(inscan, + : acc)
  for (uint64_t i = 0; i < n; ++i) {
    acc += src[i];
#pragma omp scan inclusive(acc)
    dst[i] = acc;
  }
}