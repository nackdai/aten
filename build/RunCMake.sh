#!/bin/bash

build_type=${1:-Release}

echo "Build Type is [${build_type}]"

cmake \
-D CMAKE_BUILD_TYPE=${build_type} \
-D CMAKE_CXX_COMPILER=/usr/local/clang_8.0.0/bin/clang++ \
-D CUDA_HOST_COMPILER=/usr/local/clang_8.0.0/bin/clang \
-D BUILD_UTILS=FALSE \
-D GLFW_BUILD_DOCS=FALSE \
-D GLFW_BUILD_EXAMPLES=FALSE \
-D GLFW_BUILD_TESTS=FALSE \
-L -G Ninja ..
