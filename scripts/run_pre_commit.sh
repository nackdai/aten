#!/bin/bash

set -eu
set -o pipefail

work_dir=$(realpath "${1}")

aten_dev_image="$(docker images -q aten_dev:latest 2>/dev/null)"

CONTAINER_NAME="aten_dev"
WORKSPACE="/work"

if [[ -n "${aten_dev_image}" ]]; then
  docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true

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

function finally() {
  docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 0
}

trap finally EXIT
