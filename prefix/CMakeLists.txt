add_library(prefix_scan_kernels STATIC)
target_sources(prefix_scan_kernels
  PRIVATE
  kernels.cpp
)
target_link_libraries(prefix_scan_kernels
  PRIVATE
  OpenMP::OpenMP_CXX
)
target_compile_options(prefix_scan_kernels
  PRIVATE
  -march=native
  -fsave-optimization-record
)

add_executable(prefix_scan_bench)
target_sources(prefix_scan_bench
  PRIVATE
  main.cpp
)
target_link_libraries(prefix_scan_bench 
  PRIVATE
  benchmark::benchmark 
  benchmark::benchmark_main
  prefix_scan_kernels
)