#include "predication.hpp"

#include <immintrin.h>

void gt_zero_add_avx512(uint64_t n, float *const __restrict x) {
  [[assume(n % 16 == 0)]];
  for (uint64_t i = 0; i < n; i += 16) {
    __m512 xV = _mm512_loadu_ps(x + i);
    __mmask16 mask = _mm512_cmp_ps_mask(xV, _mm512_setzero_ps(), _CMP_NLE_UQ);
    xV = _mm512_mask_add_ps(xV, mask, xV, _mm512_set1_ps(1.f));
    _mm512_storeu_ps(x + i, xV);
  }
}