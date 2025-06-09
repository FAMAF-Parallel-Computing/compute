cmake_minimum_required(VERSION 3.30)
project(stencil CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(OpenMP REQUIRED)

find_package(benchmark REQUIRED CONFIG)

enable_testing()
include(GoogleTest)
find_package(GTest REQUIRED CONFIG)

add_subdirectory(stencil)
#add_subdirectory(prefix)
add_subdirectory(sandbox)
add_subdirectory(conv)
add_subdirectory(integration)