#ifndef PREFIX_SCAN_KERNELS_HPP
#define PREFIX_SCAN_KERNELS_HPP

#include <cstdint>

void prefix_scan(uint64_t n, const float *const __restrict src,
                 float *const __restrict dst, const double init);

void prefix_scan_omp_simd(uint64_t n, const float *const __restrict src,
                          float *const __restrict dst, const double init);

#endif