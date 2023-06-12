#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [work directory]
EOF
  exit 1
}

if (("${#}" != 1)); then
  usage
fi

work_dir="$(realpath "${1}")"

aten_image="$(docker images -q aten:latest 2>/dev/null)"

CONTAINER_NAME="aten"
WORKSPACE="/work"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

trap 'kill_container ${CONTAINER_NAME}' EXIT ERR

if [[ -z "${aten_image}" ]]; then
  echo "No docker image aten_dev::latest"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  launch_docker "${CONTAINER_NAME}" "${aten_image}" \
    "-d \
    -w ${WORKSPACE} \
    --mount type=bind,src=${work_dir},target=${WORKSPACE}"

  docker exec "${CONTAINER_NAME}" bash -c "cd ${WORKSPACE}/build/aten_unittest && ./aten_unittest"
fi
