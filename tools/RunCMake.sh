#!/bin/bash

set -eu
set -o pipefail

build_type=${1:-Release}

compute_capability=${2}

CLANG_VERSION=17

echo "Build Type is [${build_type}]"
echo "Compute Capability is [${compute_capability}]"

cmake \
  -D CMAKE_BUILD_TYPE="${build_type}" \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-${CLANG_VERSION} \
  -D CMAKE_C_COMPILER=/usr/bin/clang-${CLANG_VERSION} \
  -D CUDA_HOST_COMPILER=/usr/bin/clang-${CLANG_VERSION} \
  -D CUDA_TARGET_COMPUTE_CAPABILITY="${compute_capability}" \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -L -G Ninja ..
