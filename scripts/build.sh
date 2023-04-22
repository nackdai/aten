#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options]
Options:
  -b <build_config>
  -c <compute_capability>
  -d <source_code_root_dir>
ex) ./scripts/build.sh -b Release -c 75 -d .
EOF
  exit 1
}

build_type="Release"
compute_capability="75"
work_dir="."

while getopts "b:c:d:" opt; do
  case "${opt}" in
    b)
      build_type="${OPTARG}"
      ;;
    c)
      compute_capability="${OPTARG}"
      ;;
    d)
      work_dir="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

work_dir="${1:-"."}"
work_dir="$(realpath "${work_dir}")"

if [[ ! -d "${work_dir}" ]]; then
  echo "${work_dir} doesn't exist"
  exit 1
fi

echo "Build config: [${build_type}]"
echo "Compute Capability: [${compute_capability}]"
echo "Build dir: [${work_dir}]"

aten_image="$(docker images -q aten:latest 2>/dev/null)"

CONTAINER_NAME="aten"
WORKSPACE="/work"

cmake_cmd="cmake \
  -D CMAKE_BUILD_TYPE=${build_type} \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-9 \
  -D CUDA_HOST_COMPILER=/usr/bin/clang-9 \
  -D CUDA_TARGET_COMPUTE_CAPABILITY=${compute_capability} \
  -G Ninja .."

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

if [[ -z "${aten_image}" ]]; then
  echo "No docker image aten_dev::latest"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  launch_docker "${WORKSPACE}" "${aten_image}" \
    "-d \
     -w ${WORKSPACE} \
     --mount type=bind,src=${work_dir},target=${WORKSPACE} \
     --mount type=bind,src=${PWD}/.home,target=${HOME} \
     -e HOME=${HOME}"

  docker exec "${CONTAINER_NAME}" bash -c "mkdir -p build && cd build && ${cmake_cmd}"
  docker exec "${CONTAINER_NAME}" bash -c "cd build && ninja -j 4"
fi
