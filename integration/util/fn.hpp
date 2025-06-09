#ifndef INTEGRATION_UTIL_FN_HPP
#define INTEGRATION_UTIL_FN_HPP

/*
The functions are so simple so that they can be inlined.
But I want to show that one can declare with omp simd for a single TU.
*/

float inv_sqrt(float x);

float pi_fn(float x);

#endif