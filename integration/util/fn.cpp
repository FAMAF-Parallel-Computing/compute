#include "fn.hpp"

#include <cmath>

float inv_sqrt(float x) { return 1.f / std::sqrt(x); }

float pi_fn(float x) { return 1.f / (1.f + x * x); }