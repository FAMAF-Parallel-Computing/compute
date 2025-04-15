#ifndef CONV_KERNELS_HPP
#define CONV_KERNELS_HPP

#include <cstdint>

void conv1d_3(uint64_t n, const float *const __restrict src,
              float *const __restrict dst, float *const __restrict filter);

#endif