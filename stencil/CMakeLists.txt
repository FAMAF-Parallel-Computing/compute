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
  benchmark::benchmark 
  benchmark::benchmark_main
  stencil_kernels
)
