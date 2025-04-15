#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include "kernels.hpp"

using namespace std;

static void BM_conv1d_3(benchmark::State &state) {
  const uint64_t n = state.range(0);

  auto input = make_unique<float[]>(n);
  auto kernel = make_unique<float[]>(3);
  auto output = make_unique<float[]>(n - 2);

  for (auto _ : state) {
    conv1d_3(n, input.get(), kernel.get(), output.get());
  }
}
BENCHMARK(BM_conv1d_3)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);