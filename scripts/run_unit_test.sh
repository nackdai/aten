#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 <docker_image> <base_directory>
  <docker_iamge>   : docker image to run build
  <base_directory> : Base directory to store executables
ex) $0 aten:latest ./
EOF
  exit 1
}

docker_image="${1}"
base_dir="${2}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

bash -c "${SCRIPT_DIR}/run_executable.sh -b ${base_dir} -d ${docker_image} -e aten_unittest"
