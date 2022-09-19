#!/bin/bash

set -eu
set -o pipefail

# TODO
# Usage:
# run_pre_commit.sh <dir_to_run_pre_commit>
# ex) ./scripts/run_pre_commit.sh ./

work_dir=$(realpath "${1}")

aten_dev_image="$(docker images -q aten_dev:latest 2>/dev/null)"

CONTAINER_NAME="aten_dev"
WORKSPACE="/work"

function finally() {
  docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap finally EXIT ERR

if [[ -n "${aten_dev_image}" ]]; then
  # NOTE: https://stackoverflow.com/questions/62539288/where-does-pre-commit-install-environments
  docker run -it -d \
    --name "${CONTAINER_NAME}" \
    -v "${work_dir}":"${WORKSPACE}" \
    -w "${WORKSPACE}" \
    -e PRE_COMMIT_HOME="${WORKSPACE}/.cache/pre-commit" \
    "${aten_dev_image}" >/dev/null 2>&1

  docker exec "${CONTAINER_NAME}" bash -c "git config --global --add safe.directory ${WORKSPACE}"
  docker exec "${CONTAINER_NAME}" bash -c 'eval "$(pyenv init --path)" && pre-commit run -a'
fi
