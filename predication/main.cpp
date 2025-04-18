#include <cstdint>
#include <memory>
#include <print>
#include <random>

#include "predication.hpp"

using namespace std;

int main(int argc, char **argv) {
  auto x = make_unique<float[]>(32);

  minstd_rand rand(0);
  uniform_real_distribution<float> dist(-10.f, 10.f);
  for (int i = 0; i < 32; ++i) {
    x[i] = dist(rand);
  }

  println("Before gt_zero_add_avx512:");
  for (int i = 0; i < 32; ++i) {
    if (i < 31)
      print("{:5.1f} ", x[i]);
    else
      println("{:5.1f}", x[i]);
  }

  gt_zero_add_avx512(32, x.get());

  println("After gt_zero_add_avx512:");
  for (int i = 0; i < 32; ++i) {
    if (i < 31)
      print("{:5.1f} ", x[i]);
    else
      println("{:5.1f}", x[i]);
  }

  for (int i = 0; i < 32; ++i) {
    x[i] = dist(rand);
  }
  x[0] = 144.f;

  println("Before gt_zero_sqrt_add_lt_zero_avx512:");
  for (int i = 0; i < 32; ++i) {
    if (i < 31)
      print("{:5.1f} ", x[i]);
    else
      println("{:5.1f}", x[i]);
  }
  gt_zero_sqrt_add_lt_zero_avx512(32, x.get());
  println("After gt_zero_sqrt_add_lt_zero_avx512:");
  for (int i = 0; i < 32; ++i) {
    if (i < 31)
      print("{:5.1f} ", x[i]);
    else
      println("{:5.1f}", x[i]);
  }

  return 0;
}