#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <docker_image> <work_directory>
  <docker_iamge>   : docker image to run build
  <work_directory> : Work directory. This is work directiory in docker container
ex) $0 aten:latest .
EOF
  exit 1
}

if [ "${#}" -ne 2 ]; then
  echo "Argumaents are not specified properly"
  usage
fi

docker_image="${1}"
work_dir="$(realpath "${2}")"

if [[ ! -d "${work_dir}" ]]; then
  echo "${work_dir} doesn't exist"
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

docker_image_id=$(get_image_id "${docker_image}")

CONTAINER_NAME="aten_dev"
WORKSPACE="/work"

if [[ -z "${docker_image_id}" ]]; then
  echo "No docker image ${docker_image}"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  launch_docker "${CONTAINER_NAME}" "${docker_image_id}" \
    "-w ${WORKSPACE} \
    --mount type=bind,src=${work_dir},target=${WORKSPACE} \
    --mount type=bind,src=${PWD}/.home,target=${HOME}" "bash"
fi
