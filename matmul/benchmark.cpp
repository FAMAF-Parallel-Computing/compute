#include <cstdint>
#include <memory>
#include <random>

#include <benchmark/benchmark.h>

#include "kernel.hpp"

using namespace std;

unique_ptr<float[]> generateMatrix(uint32_t n) {
  unique_ptr<float[]> matrix{new float[n * n]{}};

  random_device rd;
  minstd_rand gen(rd());
  uniform_real_distribution<float> dist(-10.f, 10.f);

  for (uint32_t i = 2; i < n - 2; ++i) {
    for (uint32_t j = 2; j < n - 2; ++j) {
      matrix[i * n + j] = dist(gen);
    }
  }

  return matrix;
}

static void BM_sgemm_baseline(benchmark::State &state) {
  const uint64_t n = state.range(0);

  auto A = generateMatrix(n);
  auto B = generateMatrix(n);
  auto C = generateMatrix(n);

  for (auto _ : state) {
    sgemm_v0(A.get(), B.get(), C.get(), n, n, n);
  }

  const double operations = n * n * n * 2.;
  state.counters["FLOPS"] = benchmark::Counter(
      operations, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_sgemm_baseline)->RangeMultiplier(2)->Range(64, 1024);

static void BM_sgemm_v1(benchmark::State &state) {
  const uint64_t n = state.range(0);

  auto A = generateMatrix(n);
  auto B = generateMatrix(n);
  auto C = generateMatrix(n);

  for (auto _ : state) {
    sgemm_v1(A.get(), B.get(), C.get(), n, n, n);
  }

  const double operations = n * n * n * 2.;
  state.counters["FLOPS"] = benchmark::Counter(
      operations, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_sgemm_baseline)->RangeMultiplier(2)->Range(64, 1024);