#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options] -- <Args to executable>
Options:
  -b <directory>    : Base directory to store executables. This option is necessary
  -e <executable>   : Name to run executable. This option is necessary
  -d <docker_image> : docker image to run executable. This option is necessary

Args to executable: Arguments to pass to executable

ex) $0 -d build -e xxx -d aten:latest -- -a
EOF
  exit 1
}

base_dir=""
executable=""
docker_image=""

while getopts "b:e:d:" opt; do
  case "${opt}" in
    b)
      base_dir="${OPTARG}"
      ;;
    e)
      executable="${OPTARG}"
      ;;
    d)
      docker_image="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

base_dir="$(realpath "${base_dir}")"

if [[ ! -d "${base_dir}" ]]; then
  echo "${base_dir} doesn't exist"
  exit 1
fi

declare -a args_to_exec=()

# If arguments are still remaining, store it for passing to specified exectuable.
if [ "${#}" -gt 0 ]; then
  args_to_exec=("${@}")
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/docker_util"

docker_image_id=$(get_image_id "${docker_image}")

CONTAINER_NAME="aten"
WORKSPACE="/work"

if [[ -z "${docker_image_id}" ]]; then
  echo "No docker image ${docker_image}"
  exit 1
else
  kill_container "${CONTAINER_NAME}"

  launch_docker "${CONTAINER_NAME}" "${docker_image_id}" \
    "-d -w ${WORKSPACE} \
    --mount type=bind,src=${base_dir},target=${WORKSPACE} \
    --mount type=bind,src=${PWD}/.home,target=${HOME} \
    -e HOME=${HOME}"

  executable_path="${WORKSPACE}/${executable}/${executable}"

  ld_library_path="${WORKSPACE}/lib"
  ld_library_path="LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${ld_library_path}"

  docker exec "${CONTAINER_NAME}" bash -c "${ld_library_path} ${executable_path} ${args_to_exec[*]}"
fi
