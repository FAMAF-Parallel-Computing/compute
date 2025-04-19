#ifndef PREDICATION_HPP

#include <cstdint>

void gt_zero_add_avx512(uint64_t n, float *const __restrict x);

void gt_zero_sqrt_add_lt_zero_avx512(uint64_t n, float *const __restrict x);

#endif