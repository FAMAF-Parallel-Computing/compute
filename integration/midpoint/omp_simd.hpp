#ifndef INTEGRATION_MIDPOINT_BASELINE_HPP
#define INTEGRATION_MIDPOINT_BASELINE_HPP

#include <cstdint>

// Mmmmm templated fn can only be defined in header files
template <typename F>
float midpoint_integration_omp(const uint32_t n, const float a, const float b,
                               F fn) {
  const float h = (b - a) / n;
  float sum = 0.f;

#pragma omp simd reduction(+ : sum)
  for (uint32_t i = 0; i < n; ++i) {
    const float x = a + (i + .5f) * h;
    sum += fn(x);
  }

  return sum * h;
}

#endif