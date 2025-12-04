cmake_minimum_required(VERSION 3.27)

# TODO: FindCUDA is deprecated. Modification to fit into new cmake takes a long
# time. To recover it temporarily, we need to specify the policy CMP0146.
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

project(aten)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_UTILS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)

# Define Release by default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "Build type not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

message("Build Type: " ${CMAKE_BUILD_TYPE})

message("CUDA_TARGET_COMPUTE_CAPABILITY: " ${CUDA_TARGET_COMPUTE_CAPABILITY})

# Get compute capability dynamicaly
# https://stackoverflow.com/questions/35485087/determining-which-gencode-compute-arch-values-i-need-for-nvcc-within-cmaks
if(NOT CUDA_TARGET_COMPUTE_CAPABILITY)
  execute_process(
    COMMAND bash -c "${CMAKE_CURRENT_SOURCE_DIR}/get_cuda_sm.sh | xargs echo -n"
    OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY)

  execute_process(
    COMMAND
      bash -c
      "echo -n ${CUDA_TARGET_COMPUTE_CAPABILITY} | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n"
    OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY)

  message("CUDA device-side code will assume compute capability "
          ${FORMATTED_COMPUTE_CAPABILITY})
endif()

add_subdirectory(3rdparty)
add_subdirectory(src)
