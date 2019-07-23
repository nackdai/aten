cmake_minimum_required(VERSION 3.10.0)

project(aten)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

find_package(CUDA REQUIRED)

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})

add_subdirectory(3rdparty)
add_subdirectory(src)
