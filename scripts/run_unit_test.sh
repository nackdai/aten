#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <docker_image> <work_directory>
  <docker_iamge>   : docker image to run build
  <work_directory> : Work directory. This is work directiory in docker container
ex) $0 aten:latest ./
EOF
  exit 1
}

docker_image="${1}"
work_dir="$(realpath "${2}")"

if [[ ! -d "${work_dir}" ]]; then
  echo "${work_dir} doesn't exist"
  exit 1
fi

CONTAINER_NAME="aten"
WORKSPACE="/work"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

docker_image_id=$(get_image_id "${docker_image}")

trap 'kill_container ${CONTAINER_NAME}' EXIT ERR

if [[ -z "${docker_image_id}" ]]; then
  echo "No docker image ${docker_image}"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  launch_docker "${CONTAINER_NAME}" "${docker_image_id}" \
    "-d \
    -w ${WORKSPACE} \
    --mount type=bind,src=${work_dir},target=${WORKSPACE}"

  docker exec "${CONTAINER_NAME}" bash -c "cd ${WORKSPACE}/build/aten_unittest && ./aten_unittest"
fi
