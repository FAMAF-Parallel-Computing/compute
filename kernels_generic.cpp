#include "kernels.hpp"

#include <memory>

#include <iostream>

using namespace std;

void stencil_1D3P_inplace(const uint64_t n, float *const __restrict inout) {
  for (uint64_t i = 1; i < n - 1; ++i) {
    inout[i] = (inout[i - 1] + inout[i] + inout[i + 1]) / 3.f;
  }
}

void stencil_1D3P_inplace_aligned(const uint64_t n,
                                  float *const __restrict inout) {
  auto *inoutA = assume_aligned<64>(inout);
  for (uint64_t i = 1; i < n - 1; ++i) {
    inoutA[i] = (inoutA[i - 1] + inoutA[i] + inoutA[i + 1]) / 3.f;
  }
}

void stencil_1D3P(const uint64_t n, const float *const __restrict in,
                  float *const __restrict out) {
  for (uint64_t i = 1; i < n - 1; ++i) {
    out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3.f;
  }
}

void stencil_1D3P_aligned(const uint64_t n, const float *const __restrict in,
                          float *const __restrict out) {
  auto *inA = assume_aligned<64>(in);
  auto *outA = assume_aligned<64>(out);
  for (uint64_t i = 1; i < n - 1; ++i) {
    outA[i] = (inA[i - 1] + inA[i] + inA[i + 1]) / 3.f;
  }
}

void stencil_2D5P_inplace(const uint64_t n, const uint64_t m,
                          float *const __restrict inout) {
  for (uint64_t i = 1; i < n - 1; ++i) {
    for (uint64_t j = 1; j < m - 1; ++j) {
      inout[i * m + j] =
          (inout[(i - 1) * m + j] + inout[i * m + j] + inout[(i + 1) * m + j] +
           inout[i * m + (j - 1)] + inout[i * m + (j + 1)]) /
          5.f;
    }
  }
}
void stencil_2D5P_inplace_aligned(const uint64_t n, const uint64_t m,
                                  float *const __restrict inout) {
  auto *inoutA = assume_aligned<64>(inout);
  for (uint64_t i = 1; i < n - 1; ++i) {
    for (uint64_t j = 1; j < m - 1; ++j) {
      inoutA[i * m + j] = (inoutA[(i - 1) * m + j] + inoutA[i * m + j] +
                           inoutA[(i + 1) * m + j] + inoutA[i * m + (j - 1)] +
                           inoutA[i * m + (j + 1)]) /
                          5.f;
    }
  }
}

void stencil_2D5P(const uint64_t n, const uint64_t m,
                  const float *const __restrict in,
                  float *const __restrict out) {
  for (uint64_t i = 1; i < n - 1; ++i) {
    for (uint64_t j = 1; j < m - 1; ++j) {
      out[i * m + j] =
          (in[(i - 1) * m + j] + in[i * m + j] + in[(i + 1) * m + j] +
           in[i * m + (j - 1)] + in[i * m + (j + 1)]) /
          5.f;
    }
  }
}

void stencil_2D5P_ji(const uint64_t n, const uint64_t m,
                     const float *const __restrict in,
                     float *const __restrict out) {
  for (uint64_t j = 1; j < m - 1; ++j) {
    for (uint64_t i = 1; i < n - 1; ++i) {
      out[i * m + j] =
          (in[(i - 1) * m + j] + in[i * m + j] + in[(i + 1) * m + j] +
           in[i * m + (j - 1)] + in[i * m + (j + 1)]) /
          5.f;
    }
  }
}

void stencil_2D5P_aligned(const uint64_t n, const uint64_t m,
                          const float *const __restrict in,
                          float *const __restrict out) {
  auto *inA = assume_aligned<64>(in);
  auto *outA = assume_aligned<64>(out);
  for (uint64_t i = 1; i < n - 1; ++i) {
    for (uint64_t j = 1; j < m - 1; ++j) {
      outA[i * m + j] =
          (inA[(i - 1) * m + j] + inA[i * m + j] + inA[(i + 1) * m + j] +
           inA[i * m + (j - 1)] + inA[i * m + (j + 1)]) /
          5.f;
    }
  }
}

void stencil_2D9P_inplace(const uint64_t n, const uint64_t m,
                          float *const __restrict inout) {
  for (uint64_t i = 2; i < n - 2; ++i) {
    for (uint64_t j = 2; j < m - 2; ++j) {
      inout[i * m + j] =
          (inout[(i - 2) * m + j] + inout[(i - 1) * m + j] + inout[i * m + j] +
           inout[(i + 1) * m + j] + inout[(i + 2) * m + j] +
           inout[i * m + (j - 2)] + inout[i * m + (j - 1)] +
           inout[i * m + (j + 1)] + inout[i * m + (j + 2)]) /
          9.f;
    }
  }
}

void stencil_2D9P_inplace_aligned(const uint64_t n, const uint64_t m,
                                  float *const __restrict inout) {
  auto *inoutA = assume_aligned<64>(inout);
  for (uint64_t i = 2; i < n - 2; ++i) {
    for (uint64_t j = 2; j < m - 2; ++j) {
      inoutA[i * m + j] = (inoutA[(i - 2) * m + j] + inoutA[(i - 1) * m + j] +
                           inoutA[i * m + j] + inoutA[(i + 1) * m + j] +
                           inoutA[(i + 2) * m + j] + inoutA[i * m + (j - 2)] +
                           inoutA[i * m + (j - 1)] + inoutA[i * m + (j + 1)] +
                           inoutA[i * m + (j + 2)]) /
                          9.f;
    }
  }
}

void stencil_2D9P(const uint64_t n, const uint64_t m,
                  const float *const __restrict in,
                  float *const __restrict out) {
  for (uint64_t i = 2; i < n - 2; ++i) {
    for (uint64_t j = 2; j < m - 2; ++j) {
      out[i * m + j] =
          (in[(i - 2) * m + j] + in[(i - 1) * m + j] + in[i * m + j] +
           in[(i + 1) * m + j] + in[(i + 2) * m + j] + in[i * m + (j - 2)] +
           in[i * m + (j - 1)] + in[i * m + (j + 1)] + in[i * m + (j + 2)]) /
          9.f;
    }
  }
}

void stencil_2D9P_aligned(const uint64_t n, const uint64_t m,
                          const float *const __restrict in,
                          float *const __restrict out) {
  auto *inA = assume_aligned<64>(in);
  auto *outA = assume_aligned<64>(out);
  for (uint64_t i = 2; i < n - 2; ++i) {
    for (uint64_t j = 2; j < m - 2; ++j) {
      outA[i * m + j] =
          (inA[(i - 2) * m + j] + inA[(i - 1) * m + j] + inA[i * m + j] +
           inA[(i + 1) * m + j] + inA[(i + 2) * m + j] + inA[i * m + (j - 2)] +
           inA[i * m + (j - 1)] + inA[i * m + (j + 1)] + inA[i * m + (j + 2)]) /
          9.f;
    }
  }
}

void stencil_2D5P_omp(const uint64_t n, const uint64_t m,
                      const float *const __restrict in,
                      float *const __restrict out) {
#pragma omp parallel for schedule(static)
  for (uint64_t i = 1; i < n - 1; ++i) {
    for (uint64_t j = 1; j < m - 1; ++j) {
      out[i * m + j] =
          (in[(i - 1) * m + j] + in[i * m + j] + in[(i + 1) * m + j] +
           in[i * m + (j - 1)] + in[i * m + (j + 1)]) /
          5.f;
    }
  }
}

void stencil_2D5P_omp_2(const uint64_t n, const uint64_t m,
                        const float *const __restrict in,
                        float *const __restrict out) {
#pragma omp parallel for collapse(2) schedule(static)
  for (uint64_t i = 1; i < n - 1; ++i) {
    for (uint64_t j = 1; j < m - 1; ++j) {
      out[i * m + j] =
          (in[(i - 1) * m + j] + in[i * m + j] + in[(i + 1) * m + j] +
           in[i * m + (j - 1)] + in[i * m + (j + 1)]) /
          5.f;
    }
  }
}