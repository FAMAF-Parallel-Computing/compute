#include <cstdint>
#include <print>

#include <immintrin.h>

using namespace std;

// if i change to the not aligned version, it gives seg fault
void permute(const uint64_t n, float *const __restrict a,
             float *const __restrict b, float *__restrict out) {
  for (uint64_t i = 0; i < n; i += 16) {
    auto aV = _mm512_loadu_ps(&a[i]);
    auto bV = _mm512_loadu_ps(&b[i]);
    auto rV =
        _mm512_permutex2var_ps(aV,
                               _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19,
                                                3, 18, 2, 17, 1, 16, 0),
                               bV);
    _mm512_storeu_ps(&out[i], rV);
  }
}

int main(int argc, char **argv) {
  auto *a = new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto *b = new float[]{100, 101, 102, 103, 104, 105, 106, 107,
                        108, 109, 110, 111, 112, 113, 114, 115};
  auto *r = new float[16];

  permute(16, a, b, r);

  for (uint64_t i = 0; i < 16; ++i) {
    println("r[{:2}] = {:6.1f}", i, r[i]);
  }

  return 0;
}