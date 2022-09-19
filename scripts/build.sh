#!/bin/bash

set -eu
set -o pipefail

# TODO
# Usage:
# build.sh <build_config> <compute_capability> <source_code_root_dir>
# ex) build.sh Release 75 .

build_type=${1:-Release}

compute_capability=${2}

echo "Build Type is [${build_type}]"
echo "Compute Capability is [${compute_capability}]"

work_dir=$(realpath "${3}")

aten_image="$(docker images -q aten:latest 2>/dev/null)"

CONTAINER_NAME="aten"
WORKSPACE="/work"

cmake_cmd="cmake \
  -D CMAKE_BUILD_TYPE=${build_type} \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-9 \
  -D CUDA_HOST_COMPILER=/usr/bin/clang-9 \
  -D CUDA_TARGET_COMPUTE_CAPABILITY=${compute_capability} \
  -L -G Ninja .."

if [[ -n "${aten_image}" ]]; then
  docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  docker run -it -d \
    --name "${CONTAINER_NAME}" \
    -v "${work_dir}":"${WORKSPACE}" \
    -w "${WORKSPACE}" \
    "${aten_image}" >/dev/null 2>&1

  docker exec "${CONTAINER_NAME}" bash -c "cd ${WORKSPACE} && mkdir -p build && cd build && ${cmake_cmd}"
  docker exec "${CONTAINER_NAME}" bash -c "cd build && ninja -j 4"
fi

function finally() {
  docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 0
}

trap finally EXIT
