#include <gtest/gtest.h>

#include <numbers>
#include <random>

#include <omp_simd.hpp>

#include <fn.hpp>

using namespace std;

#pragma omp declare simd
float pi_fn(float x);

TEST(midpointIntegrationTest, piCorrectness) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<uint32_t> dis(100, 1000);
  const auto n = dis(gen);
  const auto r = 4.f * midpoint_integration_omp(n, 0.f, 1.f, pi_fn);
  EXPECT_NEAR(r, std::numbers::pi_v<float>, 1e-5f);
}
