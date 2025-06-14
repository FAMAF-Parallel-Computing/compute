add_library(matmul STATIC)
target_sources(matmul
  PRIVATE
  kernel.cpp
  PUBLIC
  FILE_SET HEADERS
  FILES kernel.hpp
)
target_link_libraries(matmul
  PRIVATE
  OpenMP::OpenMP_CXX
)

add_executable(matmul_bench)
target_sources(matmul_bench
  PRIVATE
  benchmark.cpp
)
target_link_libraries(matmul_bench 
  PRIVATE
  benchmark::benchmark 
  benchmark::benchmark_main
  matmul
)

add_executable(
  matmul_test
)
target_sources(matmul_test
  PRIVATE
  test.cpp
)
target_link_libraries(matmul_test
  PRIVATE
  matmul
  GTest::gtest GTest::gtest_main
)
gtest_discover_tests(matmul_test)
