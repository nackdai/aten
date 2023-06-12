#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <dir_to_run_pre_commit>
ex) ./scripts/run_pre_commit.sh ./
EOF
  exit 1
}

if (("${#}" != 1)); then
  usage
fi

work_dir="$(realpath "${1}")"

aten_dev_image="$(docker images -q aten_dev:latest 2>/dev/null)"

CONTAINER_NAME="aten_dev"
WORKSPACE="/work"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

trap 'kill_container ${CONTAINER_NAME}' EXIT ERR

if [[ -z "${aten_dev_image}" ]]; then
  echo "No docker image aten_dev::latest"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  # NOTE: https://stackoverflow.com/questions/62539288/where-does-pre-commit-install-environments
  launch_docker "${CONTAINER_NAME}" "${aten_dev_image}" \
    "-d -w ${WORKSPACE} \
    --mount type=bind,src=${work_dir},target=${WORKSPACE} \
    --mount type=bind,src=${PWD}/.home,target=${HOME} \
    -e HOME=${HOME}"

  docker exec "${CONTAINER_NAME}" bash -c 'pre-commit run -a'
fi
