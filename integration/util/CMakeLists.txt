add_library(integration_util STATIC)
target_sources(integration_util
  PRIVATE
  fn.cpp
  PUBLIC
  FILE_SET HEADERS
  FILES fn.hpp
)