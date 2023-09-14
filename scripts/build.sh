#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options]
Options:
  -b <build_config>       : Build configuration. Default is "Release"
  -c <compute_capability> : Compute capability for CUDA. No need to specify ".". If it's "7.5", it's specified as "75". Default is "75"
  -d <docker_iamge>       : docker image to run build. This option is necessary
ex) $0 -b Release -c 75 -d aten:latest
EOF
  exit 1
}

build_type="Release"
compute_capability="75"
docker_image=""

while getopts "b:c:d:" opt; do
  case "${opt}" in
    b)
      build_type="${OPTARG}"
      ;;
    c)
      compute_capability="${OPTARG}"
      ;;
    d)
      docker_image="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

echo "Build config: [${build_type}]"
echo "Compute Capability: [${compute_capability}]"

cmake_cmd="cmake \
  -D CMAKE_BUILD_TYPE=${build_type} \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-12 \
  -D CMAKE_C_COMPILER=/usr/bin/clang-12 \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -D CUDA_HOST_COMPILER=/usr/bin/clang-12 \
  -D CUDA_TARGET_COMPUTE_CAPABILITY=${compute_capability} \
  -G Ninja .."

# In first running, not remove docker container for second running.
python3 ./scripts/docker_operator.py -i "${docker_image}" -c "mkdir -p build && cd build && ${cmake_cmd}"

# No need to keep docker conainer after running command. So, specify remove option.
python3 ./scripts/docker_operator.py -r -i "${docker_image}" -c "cd build && ninja -j 4"
