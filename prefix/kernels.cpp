#include "kernels.hpp"

#include <immintrin.h>

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

void prefix_scan_f32_avx512(uint64_t n, const float *__restrict src,
                            float *__restrict dst, const float init) {
  [[assume(n % 32 == 0)]];

  // We process 2 vectors of 16 floats at a time
  constexpr uint64_t V_ELEMENTS = 16;
  constexpr uint64_t N_UNROLL = 32;

  __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10,
      zmm11;
  __m512 zmmTmp1, zmmTmp2;

  __m512 zmmAcc = _mm512_set1_ps(init);
  __m512i accIdx = _mm512_set1_epi64(7);
  __m512i idx = _mm512_set1_epi64(3);

  for (uint64_t i = 0; i < n; i += N_UNROLL) {
    zmm0 = _mm512_loadu_ps(src);
    zmm5 = _mm512_loadu_ps(src + V_ELEMENTS);

    zmm2 = _mm512_maskz_permute_ps(0xAAAA, zmm0, 0b00'11'00'10);
    zmm7 = _mm512_maskz_permute_ps(0xAAAA, zmm5, 0b00'11'00'10);
    zmm10 = _mm512_add_ps(zmm0, zmm2);
    zmm11 = _mm512_add_ps(zmm5, zmm7);

    zmm1 = _mm512_maskz_permute_ps(0xCCCC, zmm0, 0b00'00'11'10);
    zmm6 = _mm512_maskz_permute_ps(0xCCCC, zmm0, 0b00'00'11'10);
    zmm10 = _mm512_add_ps(zmm10, zmm1);
    zmm11 = _mm512_add_ps(zmm11, zmm6);

    zmm1 = _mm512_maskz_permute_ps(0xCCCC, zmm1, 0b10'11'00'00);
    zmm6 = _mm512_maskz_permute_ps(0xCCCC, zmm6, 0b10'11'00'00);
    zmm10 = _mm512_add_ps(zmm10, zmm1);
    zmm11 = _mm512_add_ps(zmm11, zmm6);

    // until this point the low 256 bits works as expected

    // zmmTmp1 = _mm512_maskz_permutexvar_ps(0xFF00, idx, zmm10);

    _mm512_storeu_ps(dst, zmmTmp1);

    zmm11 = _mm512_add_ps(zmm11, zmmTmp2);
    _mm512_storeu_ps(dst + V_ELEMENTS, zmm11);

    src += N_UNROLL;
    dst += N_UNROLL;
  }
}

void prefix_scan_f64_avx512(uint64_t n, const double *__restrict src,
                            double *__restrict dst, const double init) {
  [[assume(n % 16 == 0)]];

  // We process 2 vectors of 16 floats at a time
  constexpr uint64_t V_ELEMENTS = 8;
  constexpr uint64_t N_UNROLL = 16;

  __m512d zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10,
      zmm11;
  __m512d zmmTmp1, zmmTmp2;

  __m512d zmmAcc = _mm512_set1_pd(init);
  __m512i accIdx = _mm512_set1_epi64(7);
  __m512i idx = _mm512_set1_epi64(3);

  for (uint64_t i = 0; i < n; i += N_UNROLL) {
    zmm0 = _mm512_loadu_ps(src);
    zmm5 = _mm512_loadu_ps(src + V_ELEMENTS);

    zmm2 = _mm512_maskz_permute_pd(0xAA, zmm0, 0x00);
    zmm7 = _mm512_maskz_permute_pd(0xAA, zmm5, 0x00);
    zmm10 = _mm512_add_ps(zmm0, zmm2);
    zmm11 = _mm512_add_ps(zmm5, zmm7);

    zmm1 = _mm512_maskz_permute_pd(0xCC, zmm0, 0x40);
    zmm6 = _mm512_maskz_permute_pd(0xCC, zmm0, 0b00'00'11'10);
    zmm10 = _mm512_add_ps(zmm10, zmm1);
    zmm11 = _mm512_add_ps(zmm11, zmm6);

    zmm1 = _mm512_maskz_permute_ps(0xCC, zmm1, 0x44);
    zmm6 = _mm512_maskz_permute_ps(0xCC, zmm6, 0x44);
    zmm10 = _mm512_add_ps(zmm10, zmm1);
    zmm11 = _mm512_add_ps(zmm11, zmm6);

    zmmTmp1 = _mm512_maskz_permutexvar_pd(0xF0, idx, zmm10);
    zmmTmp2 = _mm512_maskz_permutexvar_pd(0xF0, idx, zmm11);
    zmm10 = _mm512_add_pd(zmm10, zmmTmp1);
    zmm11 = _mm512_add_pd(zmm11, zmmTmp2);

    zmmTmp1 = _mm512_add_pd(zmm10, zmmAcc);
    zmmAcc = _mm512_add_pd(zmm11, zmmTmp1);
    zmmAcc = _mm512_permutexvar_pd(accIdx, zmmAcc);

    zmmTmp2 = _mm512_permutexvar_pd(accIdx, zmmTmp1);
    _mm512_storeu_pd(dst, zmmTmp1);

    zmm11 = _mm512_add_pd(zmm11, zmmTmp2);
    _mm512_storeu_pd(dst + V_ELEMENTS, zmm11);

    src += N_UNROLL;
    dst += N_UNROLL;
  }
}