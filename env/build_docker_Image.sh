#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <OPTIONS>
  -b <build_context>         : Docker build context.
  -n <nvidia_cuda_image_tag> : Base nvidia/cuda docker image
  -p <image_tag_prefix>      : Prefix for image tag
EOF
  exit 1
}

build_context="./"
image_tag_prefix=""

NVIDIA_CUDA_TAG="11.7.0-devel-ubuntu20.04"

while getopts "b:n:p:" opt; do
  case "${opt}" in
    b)
      build_context="${OPTARG}"
      ;;
    n)
      NVIDIA_CUDA_TAG="${OPTARG}"
      ;;
    p)
      :
      image_tag_prefix="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

build_context="$(realpath "${build_context}")"

nvidia_cuda="nvidia/cuda:${NVIDIA_CUDA_TAG}"
nvidia_cudagl="nvidia/cudagl:${NVIDIA_CUDA_TAG}"

docker build -t "${image_tag_prefix}${nvidia_cudagl}" --build-arg from="${nvidia_cuda}" -f "${build_context}/cudagl/Dockerfile" "${build_context}"
docker build -t "${image_tag_prefix}aten" --build-arg from="${image_tag_prefix}${nvidia_cudagl}" -f "${build_context}/aten/Dockerfile" "${build_context}"
docker build -t "${image_tag_prefix}aten_dev" --build-arg from="${image_tag_prefix}aten:latest" -f "${build_context}/dev/Dockerfile" "${build_context}"
