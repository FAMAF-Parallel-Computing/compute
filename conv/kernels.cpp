#include "kernels.hpp"

void conv1d_3(uint64_t n, const float *const __restrict src,
              float *const __restrict dst, float *const __restrict filter) {
  for (uint64_t i = 0; i < n - 2; ++i) {
    for (uint64_t j = 0; j < 3; ++j) {
      dst[i] += src[i + j] * filter[j];
    }
  }
}