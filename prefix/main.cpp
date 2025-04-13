#include <memory>

#include <benchmark/benchmark.h>

#include "kernels.hpp"

using namespace std;

static void BM_prefix_scan(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0)]};
  unique_ptr<float[]> out{new float[state.range(0)]};

  for (auto _ : state) {
    prefix_scan(state.range(0), in.get(), out.get(), 0.0);
  }
}
BENCHMARK(BM_prefix_scan)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

static void BM_prefix_scan_aligned(benchmark::State &state) {
  unique_ptr<float[]> in{new (align_val_t{64}) float[state.range(0)]};
  unique_ptr<float[]> out{new (align_val_t{64}) float[state.range(0)]};

  for (auto _ : state) {
    prefix_scan(state.range(0), in.get(), out.get(), 0.0);
  }
}
BENCHMARK(BM_prefix_scan_aligned)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

static void BM_prefix_scan_omp_simd(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0)]};
  unique_ptr<float[]> out{new float[state.range(0)]};

  for (auto _ : state) {
    prefix_scan_omp_simd(state.range(0), in.get(), out.get(), 0.0);
  }
}
BENCHMARK(BM_prefix_scan_omp_simd)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

static void BM_prefix_scan_omp_simd_aligned(benchmark::State &state) {
  unique_ptr<float[]> in{new (align_val_t{64}) float[state.range(0)]};
  unique_ptr<float[]> out{new (align_val_t{64}) float[state.range(0)]};

  for (auto _ : state) {
    prefix_scan_omp_simd(state.range(0), in.get(), out.get(), 0.0);
  }
}
BENCHMARK(BM_prefix_scan_omp_simd_aligned)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 15);
