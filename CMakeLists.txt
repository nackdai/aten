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

message("Build Type: " ${CMAKE_BUILD_TYPE})

# Get compute capability dynamicaly
# https://stackoverflow.com/questions/35485087/determining-which-gencode-compute-arch-values-i-need-for-nvcc-within-cmaks
if (NOT CUDA_TARGET_COMPUTE_CAPABILITY)
  execute_process(COMMAND 
    bash -c "${CMAKE_CURRENT_SOURCE_DIR}/get_cuda_sm.sh | xargs echo -n"
    OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY)

  execute_process(COMMAND 
    bash -c "echo -n ${CUDA_TARGET_COMPUTE_CAPABILITY} | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n"
    OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY)

  message("CUDA device-side code will assume compute capability " 
    ${FORMATTED_COMPUTE_CAPABILITY})
endif()

add_subdirectory(3rdparty)
add_subdirectory(src)
