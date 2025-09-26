#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <docker_image>
  <docker_image>   : docker image to run
ex) $0 aten:latest
EOF
  exit 1
}

if [ "${#}" -ne 1 ]; then
  echo "Arguments are not specified properly"
  usage
fi

docker_image="${1}"

python3 ./tools/docker_operator.py -r -e -i "${docker_image}"
