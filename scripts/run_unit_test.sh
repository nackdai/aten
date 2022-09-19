#!/bin/bash

set -eu
set -o pipefail

# TODO
# Usage:
# run_unit_test.sh
# ex) ./scripts/run_unit_test.sh

aten_image="$(docker images -q aten:latest 2>/dev/null)"

CONTAINER_NAME="aten"
WORKSPACE="/work"

function finally() {
  docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap finally EXIT ERR

if [[ -n "${aten_image}" ]]; then
  # NOTE: https://stackoverflow.com/questions/62539288/where-does-pre-commit-install-environments
  docker run -it -d \
    --name "${CONTAINER_NAME}" \
    -v "${PWD}":"${WORKSPACE}" \
    -w "${WORKSPACE}" \
    "${aten_image}" >/dev/null 2>&1

  docker exec "${CONTAINER_NAME}" bash -c "cd ${WORKSPACE}/build/aten_unittest && ./aten_unittest"
fi
