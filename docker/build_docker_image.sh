#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <OPTIONS>
  -b <build_context>         : Docker build context.
  -c <cuda_version>          : CUDA version. Default is 12.5.1
  -u <ubuntu_version>        : Ubuntu version. Default is 22.04
  -p <image_tag_prefix>      : Prefix for image tag
  --clear_cache              : Build with clearing cache.

ex) $0 -b ./docker -p "ghcr.io/nackdai/aten" -c 12.5.1 -u 22.04
EOF
  exit 1
}

build_context="./"
image_tag_prefix=""
docker_build_options=()

CUDA_VERSION="12.5.1"
UBUNTU_VERSION="22.04"

CUDAGL_BUILD_DIR="3rdparty/nvidia_container_image_cuda"

while getopts "b:p:c:u:-:" opt; do
  case "${opt}" in
    -)
      case "${OPTARG}" in
        clear_cache)
          docker_build_options+=("--no-cache")
          ;;
      esac
      ;;
    b)
      build_context="${OPTARG}"
      ;;
    p)
      image_tag_prefix="${OPTARG}"
      ;;
    c)
      CUDA_VERSION="${OPTARG}"
      ;;
    u)
      UBUNTU_VERSION="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

build_context="$(realpath "${build_context}")"

# Remove trailing "/" from image_tag_prefix if present.
image_tag_prefix="${image_tag_prefix%/}"

export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

base_cudagl_image_tag="${image_tag_prefix}/nvidia/cudagl"

pushd "${CUDAGL_BUILD_DIR}"
./build.sh -d --image-name "${base_cudagl_image_tag}" --cuda-version "${CUDA_VERSION}" --os ubuntu --os-version "${UBUNTU_VERSION}" --arch x86_64 --cudagl
popd

base_cudagl_image_tag="${base_cudagl_image_tag}:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"

docker build "${docker_build_options[@]}" -t "${image_tag_prefix}/aten" --build-arg from="${base_cudagl_image_tag}" -f "${build_context}/aten/Dockerfile" "${build_context}"
docker build "${docker_build_options[@]}" -t "${image_tag_prefix}/aten_dev" --build-arg from="${image_tag_prefix}/aten:latest" -f "${build_context}/dev/Dockerfile" "${build_context}"
