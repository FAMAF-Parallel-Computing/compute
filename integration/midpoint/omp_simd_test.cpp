#include <gtest/gtest.h>

#include <numbers>

#include <omp_simd.hpp>

#include <fn.hpp>

using namespace std;

#pragma omp declare simd
float pi_fn(float x);

TEST(midpointIntegrationTest, piCorrectness) {
  const auto r = 4.f * midpoint_integration_omp(1'000, 0.f, 1.f, pi_fn);
  EXPECT_NEAR(r, std::numbers::pi_v<float>, 1e-5f);
}
