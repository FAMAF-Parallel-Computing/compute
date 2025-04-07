cmake_minimum_required(VERSION 3.30)
project(stencil CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(OpenMP REQUIRED)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(benchmark)

add_library(stencil_kernels STATIC)
target_sources(stencil_kernels
  PRIVATE
  kernels_generic.cpp
)
target_link_libraries(stencil_kernels
  PRIVATE
  OpenMP::OpenMP_CXX
)
target_compile_options(stencil_kernels
  PRIVATE
  -march=native
)

add_executable(stencil_bench)
target_sources(stencil_bench
  PRIVATE
  main.cpp
)
target_link_libraries(stencil_bench 
  PRIVATE
  benchmark 
  benchmark_main
  stencil_kernels
)

add_subdirectory(sandbox)
