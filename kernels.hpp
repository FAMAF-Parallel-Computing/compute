#ifndef STENCIL_KERNELS_HPP
#define STENCIL_KERNELS_HPP

#include <cstdint>

void stencil_1D3P_inplace(const uint64_t n, float *const __restrict inout);

void stencil_1D3P_inplace_aligned(const uint64_t n,
                                  float *const __restrict inout);

void stencil_1D3P(const uint64_t n, float *const __restrict in,
                  float *const __restrict out);

void stencil_1D3P_aligned(const uint64_t n, float *const __restrict in,
                          float *const __restrict out);

void stencil_2D5P_inplace(const uint64_t n, const uint64_t m,
                          float *const __restrict inout);

void stencil_2D5P_inplace_aligned(const uint64_t n, const uint64_t m,
                                  float *const __restrict inout);

void stencil_2D5P(const uint64_t n, const uint64_t m,
                  float *const __restrict in, float *const __restrict out);

void stencil_2D5P_aligned(const uint64_t n, const uint64_t m,
                          float *const __restrict in,
                          float *const __restrict out);

void stencil_2D9P_inplace(const uint64_t n, const uint64_t m,
                          float *const __restrict inout);

void stencil_2D9P_inplace_aligned(const uint64_t n, const uint64_t m,
                                  float *const __restrict inout);

void stencil_2D9P(const uint64_t n, const uint64_t m,
                  float *const __restrict in, float *const __restrict out);

void stencil_2D9P_aligned(const uint64_t n, const uint64_t m,
                          float *const __restrict in,
                          float *const __restrict out);

#endif