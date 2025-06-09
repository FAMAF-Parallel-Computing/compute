#ifndef INTEGRATION_MIDPOINT_BASELINE_HPP
#define INTEGRATION_MIDPOINT_BASELINE_HPP

#include <cstdint>

template <typename F>
float midpoint_integration_baseline(const uint32_t n, const float a,
                                    const float b, F fn) {
  const float h = (b - a) / n;
  float sum = 0.f;

  for (uint32_t i = 0; i < n; ++i) {
    const float x = a + (i + .5f) * h;
    sum += fn(x);
  }

  return sum * h;
}

#endif