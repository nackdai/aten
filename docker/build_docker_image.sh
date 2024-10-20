#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <OPTIONS>
  -b <build_context>         : Docker build context.
  -n <nvidia_cuda_image_tag> : Base nvidia/cuda docker image
  -p <image_tag_prefix>      : Prefix for image tag
  -c                         : Build with cleaning cache.
ex) $0 -b ./docker -p foo
EOF
  exit 1
}

build_context="./"
image_tag_prefix=""
dcoker_build_opitons=()

NVIDIA_CUDA_TAG="11.7.1-devel-ubuntu20.04"

while getopts "b:n:p:c" opt; do
  case "${opt}" in
    b)
      build_context="${OPTARG}"
      ;;
    n)
      NVIDIA_CUDA_TAG="${OPTARG}"
      ;;
    p)
      image_tag_prefix="${OPTARG}"
      ;;
    c)
      dcoker_build_opitons+=("--no-cache")
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

build_context="$(realpath "${build_context}")"

# NOTE:
# Need space between ":" and "-1"
tail_image_tag_prefix="${image_tag_prefix: -1}"

# If tail character of image_tag_prefix is "/", remove it.
if [[ "${tail_image_tag_prefix}" == "/" ]]; then
  image_tag_prefix="${image_tag_prefix/%?/}"
fi

nvidia_cuda="docker.io/nvidia/cuda:${NVIDIA_CUDA_TAG}"
nvidia_cudagl="nvidia/cudagl:${NVIDIA_CUDA_TAG}"

docker pull "${nvidia_cuda}"

export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

docker build "${dcoker_build_opitons[@]}" -t "${image_tag_prefix}/${nvidia_cudagl}" --build-arg from="${nvidia_cuda}" -f "${build_context}/cudagl/Dockerfile" "${build_context}"
docker build "${dcoker_build_opitons[@]}" -t "${image_tag_prefix}/aten" --build-arg from="${image_tag_prefix}/${nvidia_cudagl}" -f "${build_context}/aten/Dockerfile" "${build_context}"
docker build "${dcoker_build_opitons[@]}" -t "${image_tag_prefix}/aten_dev" --build-arg from="${image_tag_prefix}/aten:latest" -f "${build_context}/dev/Dockerfile" "${build_context}"
