#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options]
Options:
  -b <build_config>       : Build configuration. Default is "Release"
  -c <compute_capability> : Compute capability for CUDA. No need to specify ".". If it's "7.5", it's specified as "75". Default is "75"
  -w <work_directory>     : Work directory. This is work direction in docker container. Default is "."
  -d <docker_iamge>       : docker image to run build. This option is necessary
ex) $0 -b Release -c 75 -w . -d aten:latest
EOF
  exit 1
}

build_type="Release"
compute_capability="75"
work_dir="."
docker_image=""

while getopts "b:c:w:d:" opt; do
  case "${opt}" in
    b)
      build_type="${OPTARG}"
      ;;
    c)
      compute_capability="${OPTARG}"
      ;;
    w)
      work_dir="${OPTARG}"
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

work_dir="$(realpath "${work_dir}")"

if [[ ! -d "${work_dir}" ]]; then
  echo "${work_dir} doesn't exist"
  exit 1
fi

echo "Build config: [${build_type}]"
echo "Compute Capability: [${compute_capability}]"
echo "Build dir: [${work_dir}]"

CONTAINER_NAME="aten"
WORKSPACE="${work_dir}"

cmake_cmd="cmake \
  -D CMAKE_BUILD_TYPE=${build_type} \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-12 \
  -D CUDA_HOST_COMPILER=/usr/bin/clang-12 \
  -D CUDA_TARGET_COMPUTE_CAPABILITY=${compute_capability} \
  -G Ninja .."

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

trap 'kill_container ${CONTAINER_NAME}' EXIT ERR

docker_image_id=$(get_image_id "${docker_image}")

if [[ -z "${docker_image_id}" ]]; then
  echo "No docker image ${docker_image}"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  launch_docker "${CONTAINER_NAME}" "${docker_image_id}" \
    "-d \
     -w ${WORKSPACE} \
     --mount type=bind,src=${work_dir},target=${WORKSPACE} \
     --mount type=bind,src=${PWD}/.home,target=${HOME} \
     -e HOME=${HOME}"

  docker exec "${CONTAINER_NAME}" bash -c "mkdir -p build && cd build && ${cmake_cmd}"
  docker exec "${CONTAINER_NAME}" bash -c "cd build && ninja -j 4"
fi
