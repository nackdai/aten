#!/bin/bash

set -eu
set -o pipefail

# TODO
# Usage:
# build_docker_iamge.sh <docker_file_stored_dir>
# ex) ./env/build_docker_iamge.sh ./env

function finally() {
  popd
  exit 0
}

SCRIPT_DIR=$(
  cd "$(dirname "${0}")"
  pwd
)

pushd "${SCRIPT_DIR}"

docker build -t aten ./aten
docker build -t aten_dev --build-arg base_from=aten:latest -f ./dev/Dockerfile .

trap finally EXIT
