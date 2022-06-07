#!/bin/bash

set -eu
set -o pipefail

function finally() {
    popd
    exit 0
}

SCRIPT_DIR=$(cd $(dirname $0); pwd)

pushd "${SCRIPT_DIR}"

docker build -t aten ./aten
docker build -t aten_dev --build-arg base_from=aten:latest -f ./dev/Dockerfile .

trap finally EXIT
