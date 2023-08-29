#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <docker_image> <dir_to_run_pre_commit>
  <docker_iamge>          : docker image to run build
  <dir_to_run_pre_commit> : Directory to run pre_commit
ex) $0 aten_dev:latest ./
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

CONTAINER_NAME="aten_dev"
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

  # NOTE:
  # https://stackoverflow.com/questions/62539288/where-does-pre-commit-install-environments
  launch_docker "${CONTAINER_NAME}" "${docker_image_id}" \
    "-d -w ${WORKSPACE} \
    --mount type=bind,src=${work_dir},target=${WORKSPACE} \
    --mount type=bind,src=${PWD}/.home,target=${HOME} \
    -e HOME=${HOME}"

  docker exec "${CONTAINER_NAME}" bash -c 'pre-commit run -a'
fi
