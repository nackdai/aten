#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <docker_image>
  <docker_image>          : docker image to run build
ex) $0 aten_dev:latest
EOF
  exit 1
}

if [ "${#}" -ne 1 ]; then
  echo "Argumaents are not specified properly"
  usage
fi

docker_image="${1}"

python3 ./tools/docker_operator.py -r -i "${docker_image}" -c "pre-commit run -a"
