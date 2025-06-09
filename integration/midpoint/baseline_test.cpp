#include <gtest/gtest.h>

#include <numbers>

#include <baseline.hpp>

#include <fn.hpp>

using namespace std;

TEST(midpointIntegrationTest, piCorrectness) {
  const auto r = 4.f * midpoint_integration_baseline(1'000, 0.f, 1.f, pi_fn);
  EXPECT_NEAR(r, std::numbers::pi_v<float>, 1e-5f);
}
