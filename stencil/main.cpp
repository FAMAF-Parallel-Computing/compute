#include <memory>

#include <benchmark/benchmark.h>

#include "kernels.hpp"

using namespace std;

static void BM_stencil_1D3P_inplace(benchmark::State &state) {
  unique_ptr<float[]> inout(new float[state.range(0)]);

  for (auto _ : state) {
    stencil_1D3P_inplace(state.range(0), inout.get());
  }
}
BENCHMARK(BM_stencil_1D3P_inplace)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

static void BM_stencil_1D3P_inplace_aligned(benchmark::State &state) {
  unique_ptr<float[]> inout{new (align_val_t{64}) float[state.range(0)]};

  for (auto _ : state) {
    stencil_1D3P_inplace_aligned(state.range(0), inout.get());
  }
}
BENCHMARK(BM_stencil_1D3P_inplace_aligned)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 15);

static void BM_stencil_1D3P(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0)]};
  unique_ptr<float[]> out{new float[state.range(0)]};

  for (auto _ : state) {
    stencil_1D3P(state.range(0), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_1D3P)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

static void BM_stencil_1D3P_aligned(benchmark::State &state) {
  unique_ptr<float[]> in{new (align_val_t{64}) float[state.range(0)]};
  unique_ptr<float[]> out{new (align_val_t{64}) float[state.range(0)]};

  for (auto _ : state) {
    stencil_1D3P_aligned(state.range(0), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_1D3P_aligned)->RangeMultiplier(2)->Range(1 << 8, 1 << 15);

static void BM_stencil_2D5P_inplace(benchmark::State &state) {
  unique_ptr<float[]> inout(new float[state.range(0) * state.range(1)]);

  for (auto _ : state) {
    stencil_2D5P_inplace(state.range(0), state.range(1), inout.get());
  }
}
BENCHMARK(BM_stencil_2D5P_inplace)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12});

static void BM_stencil_2D5P_inplace_aligned(benchmark::State &state) {
  unique_ptr<float[]> inout{
      new (align_val_t{64}) float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D5P_inplace_aligned(state.range(0), state.range(1), inout.get());
  }
}
BENCHMARK(BM_stencil_2D5P_inplace_aligned)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12});

static void BM_stencil_2D5P(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{new float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D5P(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D5P)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 13, 1 << 13});

static void BM_stencil_2D5P_ji(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{new float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D5P_ji(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D5P_ji)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 13, 1 << 13});

static void BM_stencil_2D5P_aligned(benchmark::State &state) {
  unique_ptr<float[]> in{
      new (align_val_t{64}) float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{
      new (align_val_t{64}) float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D5P_aligned(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D5P_aligned)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 13, 1 << 13});

static void BM_stencil_2D9P_inplace(benchmark::State &state) {
  unique_ptr<float[]> inout(new float[state.range(0) * state.range(1)]);

  for (auto _ : state) {
    stencil_2D9P_inplace(state.range(0), state.range(1), inout.get());
  }
}
BENCHMARK(BM_stencil_2D9P_inplace)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12});

static void BM_stencil_2D9P_inplace_aligned(benchmark::State &state) {
  unique_ptr<float[]> inout{
      new (align_val_t{64}) float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D9P_inplace_aligned(state.range(0), state.range(1), inout.get());
  }
}
BENCHMARK(BM_stencil_2D9P_inplace_aligned)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12});

static void BM_stencil_2D9P(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{new float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D9P(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D9P)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12});

static void BM_stencil_2D9P_aligned(benchmark::State &state) {
  unique_ptr<float[]> in{
      new (align_val_t{64}) float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{
      new (align_val_t{64}) float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D9P_aligned(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D9P_aligned)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12});

static void BM_stencil_2D5P_omp(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{new float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D5P_omp(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D5P_omp)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 13, 1 << 13})
    ->Args({1 << 14, 1 << 14})
    ->Args({1 << 15, 1 << 15})
    ->Args({1 << 16, 1 << 16})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

static void BM_stencil_2D5P_omp_2(benchmark::State &state) {
  unique_ptr<float[]> in{new float[state.range(0) * state.range(1)]};
  unique_ptr<float[]> out{new float[state.range(0) * state.range(1)]};

  for (auto _ : state) {
    stencil_2D5P_omp_2(state.range(0), state.range(1), in.get(), out.get());
  }
}
BENCHMARK(BM_stencil_2D5P_omp_2)
    ->Args({1 << 8, 1 << 8})
    ->Args({1 << 9, 1 << 9})
    ->Args({1 << 10, 1 << 10})
    ->Args({1 << 11, 1 << 11})
    ->Args({1 << 12, 1 << 12})
    ->Args({1 << 13, 1 << 13})
    ->Args({1 << 14, 1 << 14})
    ->Args({1 << 15, 1 << 15})
    ->Args({1 << 16, 1 << 16})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_MAIN();
