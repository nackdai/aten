#!/bin/bash

set -eu
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

work_dir="$(realpath "${1}")"

aten_dev_image="$(docker images -q aten:latest 2>/dev/null)"

CONTAINER_NAME="aten_dev"
WORKSPACE="/work"

kill_container "${CONTAINER_NAME}"

launch_docker "${CONTAINER_NAME}" "${aten_dev_image}" \
  "-w ${WORKSPACE} \
  --mount type=bind,src=${work_dir},target=${WORKSPACE} \
  --mount type=bind,src=${PWD}/.home,target=${HOME}" "bash"
