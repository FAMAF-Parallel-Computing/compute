#include <memory>
#include <random>

#include <gtest/gtest.h>

#include "kernel.hpp"

using namespace std;

unique_ptr<float[]> generateMatrix(uint32_t n) {
  unique_ptr<float[]> matrix{new float[n * n]{}};

  random_device rd;
  minstd_rand gen(rd());
  uniform_real_distribution<float> dist(-4.f, 4.f);

  for (uint32_t i = 2; i < n - 2; ++i) {
    for (uint32_t j = 2; j < n - 2; ++j) {
      matrix[i * n + j] = dist(gen);
    }
  }

  return matrix;
}

TEST(matmulTest_v1, correctness) {
  auto A = generateMatrix(1024);
  auto B = generateMatrix(1024);
  auto CExpected = make_unique<float[]>(1024 * 1024);
  auto C = make_unique<float[]>(1024 * 1024);

  sgemm_v0(A.get(), B.get(), CExpected.get(), 1024, 1024, 1024);
  sgemm_v1(A.get(), B.get(), C.get(), 1024, 1024, 1024);

  for (uint32_t i = 0; i < 1024; ++i) {
    for (uint32_t j = 0; j < 1024; ++j) {
      EXPECT_NEAR(CExpected[i * 1024 + j], C[i * 1024 + j], 1e-5);
    }
  }
}

TEST(matmulTest_v3, correctness) {
  auto A = generateMatrix(1024);
  auto B = generateMatrix(1024);
  auto CExpected = make_unique<float[]>(1024 * 1024);
  auto C = make_unique<float[]>(1024 * 1024);

  sgemm_v0(A.get(), B.get(), CExpected.get(), 1024, 1024, 1024);
  sgemm_v3(A.get(), B.get(), C.get(), 1024, 1024, 1024);

  for (uint32_t i = 0; i < 1024; ++i) {
    for (uint32_t j = 0; j < 1024; ++j) {
      EXPECT_NEAR(CExpected[i * 1024 + j], C[i * 1024 + j], 1e-5);
    }
  }
}

TEST(matmulTest_v4_1dtiling, correctness) {
  auto A = generateMatrix(1024);
  auto B = generateMatrix(1024);
  auto CExpected = make_unique<float[]>(1024 * 1024);
  auto C = make_unique<float[]>(1024 * 1024);

  sgemm_v0(A.get(), B.get(), CExpected.get(), 1024, 1024, 1024);
  sgemm_v4_1dtiling(A.get(), B.get(), C.get(), 1024, 1024, 1024);

  for (uint32_t i = 0; i < 1024; ++i) {
    for (uint32_t j = 0; j < 1024; ++j) {
      EXPECT_NEAR(CExpected[i * 1024 + j], C[i * 1024 + j], 1e-5);
    }
  }
}

TEST(matmulTest_v5_omp, correctness) {
  auto A = generateMatrix(1024);
  auto B = generateMatrix(1024);
  auto CExpected = make_unique<float[]>(1024 * 1024);
  auto C = make_unique<float[]>(1024 * 1024);

  sgemm_v0(A.get(), B.get(), CExpected.get(), 1024, 1024, 1024);
  sgemm_v5_omp(A.get(), B.get(), C.get(), 1024, 1024, 1024);

  for (uint32_t i = 0; i < 1024; ++i) {
    for (uint32_t j = 0; j < 1024; ++j) {
      EXPECT_NEAR(CExpected[i * 1024 + j], C[i * 1024 + j], 1e-5);
    }
  }
}