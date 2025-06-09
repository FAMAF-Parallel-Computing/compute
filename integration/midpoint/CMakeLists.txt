add_library(midpoint_baseline INTERFACE)
target_sources(midpoint_baseline
  PUBLIC
  FILE_SET HEADERS
  FILES baseline.hpp
)

add_library(midpoint_omp_simd INTERFACE)
target_sources(midpoint_omp_simd
  PUBLIC
  FILE_SET HEADERS
  FILES omp_simd.hpp
)

add_executable(
  midpoint_baseline_test
)
target_sources(midpoint_baseline_test
  PRIVATE
  baseline_test.cpp
)
target_link_libraries(midpoint_baseline_test
  PRIVATE
  integration_util
  midpoint_baseline
  GTest::gtest GTest::gtest_main
)
gtest_discover_tests(midpoint_baseline_test)

add_executable(
  midpoint_omp_simd_test
)
target_sources(midpoint_omp_simd_test
  PRIVATE
  omp_simd_test.cpp
)
target_link_libraries(midpoint_omp_simd_test
  PRIVATE
  OpenMP::OpenMP_CXX
  integration_util
  midpoint_omp_simd
  GTest::gtest GTest::gtest_main
)
gtest_discover_tests(midpoint_omp_simd_test)