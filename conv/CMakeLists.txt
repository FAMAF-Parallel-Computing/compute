add_library(conv_kernels STATIC)
target_sources(conv_kernels
  PRIVATE
  kernels.cpp
)
target_link_libraries(conv_kernels
  PRIVATE
  OpenMP::OpenMP_CXX
)
target_compile_options(conv_kernels
  PRIVATE
  -march=native
  -fsave-optimization-record
)

add_executable(conv_bench)
target_sources(conv_bench
  PRIVATE
  main.cpp
)
target_link_libraries(conv_bench 
  PRIVATE
  benchmark::benchmark 
  benchmark::benchmark_main
  conv_kernels
)