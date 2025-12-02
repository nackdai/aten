#!/bin/bash

set -eu
set -o pipefail

CUDA_VERSION="12.5.1"
UBUNTU_VERSION="22.04"

TAG="latest"

usage() {
  cat <<EOF 1>&2
Usage: $0 <OPTIONS>
  -b <build_context>      : Docker build context.
  -c <cuda_version>       : CUDA version. Default is "${CUDA_VERSION}"
  -u <ubuntu_version>     : Ubuntu version. Default is "${UBUNTU_VERSION}"
  -p <image_tag_prefix>   : Prefix for image tag
  -t <tag>                : Specify tag. Default is "${TAG}"
  --no-cache              : Build with --no-cache option.

ex) $0 -b ./docker -p "ghcr.io/nackdai/aten" -c 12.5.1 -u 22.04
EOF
  exit 1
}

build_context="./"
image_tag_prefix=""
docker_build_options=()

CUDAGL_BUILD_DIR="3rdparty/nvidia_container_image_cuda"

while getopts "b:p:c:u:t:-:f" opt; do
  case "${opt}" in
    -)
      case "${OPTARG}" in
        no-cache)
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
    t)
      TAG="${OPTARG}"
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

base_cudagl_image_prefix="${image_tag_prefix}/nvidia/cudagl"
base_cudagl_image_tag="${base_cudagl_image_prefix}:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"

if docker image inspect "${base_cudagl_image_tag}" >/dev/null 2>/dev/null; then
   echo "${base_cudagl_image_tag} is already available"
else
   # Try to pull the docker image. If it's not available, building it happens.
   # So, even if it's failed, to continue to the following sequence, need to suppress to raise an error.
   docker pull "${base_cudagl_image_tag}" >/dev/null 2>/dev/null || true
fi

pushd "${CUDAGL_BUILD_DIR}"
./build.sh -d --image-name "${base_cudagl_image_prefix}" --cuda-version "${CUDA_VERSION}" --os ubuntu --os-version "${UBUNTU_VERSION}" --arch x86_64 --cudagl
popd

docker build "${docker_build_options[@]}" -t "${image_tag_prefix}/aten:${TAG}" --build-arg from="${base_cudagl_image_tag}" -f "${build_context}/aten/Dockerfile" "${build_context}"
docker build "${docker_build_options[@]}" -t "${image_tag_prefix}/aten_dev:${TAG}" --build-arg from="${image_tag_prefix}/aten:latest" -f "${build_context}/dev/Dockerfile" "${build_context}"
