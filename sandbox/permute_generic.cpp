#include <cstdint>
#include <print>

using namespace std;

using float16 = float __attribute__((vector_size(16 * sizeof(float))));

void permute(const uint64_t n, float *const __restrict a,
             float *const __restrict b, float *__restrict out) {
  for (uint64_t i = 0; i < n; i += 16) {
    auto aV = *reinterpret_cast<float16 *>(&a[i]);
    auto bV = *reinterpret_cast<float16 *>(&b[i]);
    auto rV = __builtin_shufflevector(aV, bV, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20,
                                      5, 21, 6, 22, 7, 23);
    *reinterpret_cast<float16 *>(&out[i]) = rV;
  }
}

int main(int argc, char **argv) {
  // ... without alginment gives seg fault
  // it seems that only generates code for aligned version
  auto *a = new (align_val_t{64}) float[]{0, 1, 2,  3,  4,  5,  6,  7,
                                          8, 9, 10, 11, 12, 13, 14, 15};
  auto *b =
      new (align_val_t{64}) float[]{100, 101, 102, 103, 104, 105, 106, 107,
                                    108, 109, 110, 111, 112, 113, 114, 115};
  auto *r = new (align_val_t{64}) float[16];

  permute(16, a, b, r);

  for (uint64_t i = 0; i < 16; ++i) {
    println("r[{:2}] = {:6.1f}", i, r[i]);
  }

  return 0;
}