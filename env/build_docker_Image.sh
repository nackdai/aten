#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <OPTIONS>
  -b <build_context>     : Docker build context.
  -n <nvidia_cuda_image> : Base nvidia/cuda docker image
  -t <nvidia_cudagl_tag> : Created cudagl image tag
EOF
  exit 1
}

NVIDIA_CUDA_TAG="11.7.0-devel-ubuntu20.04"

build_context="./"
nvidia_cuda="nvidia/cuda:${NVIDIA_CUDA_TAG}"
nvidia_cudagl_tag="nvidia/cudagl:${NVIDIA_CUDA_TAG}"

while getopts "b:n:t:" opt; do
  case "${opt}" in
    b)
      build_context="${OPTARG}"
      ;;
    n)
      nvidia_cuda="${OPTARG}"
      ;;
    t)
      nvidia_cudagl_tag="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

build_context="$(realpath "${build_context}")"

docker build -t "${nvidia_cudagl_tag}" --build-arg from="${nvidia_cuda}" -f "${build_context}/cudagl/Dockerfile" "${build_context}"
docker build -t aten --build-arg from="${nvidia_cudagl_tag}" -f "${build_context}/aten/Dockerfile" "${build_context}"
docker build -t aten_dev --build-arg from=aten:latest -f "${build_context}/dev/Dockerfile" "${build_context}"
