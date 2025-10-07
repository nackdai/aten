#!/bin/bash

set -eu
set -o pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

docker_image=""
header_filter="${PWD}/src"
will_fix=false
fail_fast=false
cuda_arch="75"
target_dir=""
dir_to_compile_commands_json="${THIS_DIR}/../build"
git_diff_commits=()

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options]
Options:
  -d <docker_image>     : docker image to run build. Required.
  -h <header_filter>    : Header filter. If nothing is specified, "src" is specified.
  -g <commit> <commit>  : Specify git diff files as clang-tidy target files.
  -t <target_dir>       : Specify target directory.
                          If nothing is specified, all files in compile_commands.json are targeted.
  -c <cuda_arch>        : Specify CUDA architecture.
                          If nothing is specified, "${cuda_arch}(=7.5)" is specified.
  -p <compile_commands.json located directory> : Specify directory to locate compile_commands.json.
                                                 Default is "${dir_to_compile_commands_json}"
  -f                    : Fix code.
  --fail_fast           : If the error happens, stop immediately
ex) $0 -d aten_dev:latest -g -h ${PWD}/src --fail_fast
EOF
  exit 1
}

CONTAINER_NAME=""
kill_container() {
  local container_name="${1}"
  docker kill "${container_name}" >/dev/null 2>&1 || true
  docker container rm "${container_name}" >/dev/null 2>&1 || true
}

#trap 'kill_container ${CONTAINER_NAME}' EXIT ERR

# NOTE:
# Long option
# https://qiita.com/akameco/items/0e932d8ec372b87ccb34

while getopts "d:h:t:c:p:gf-:" opt; do
  case "${opt}" in
    -)
      case "${OPTARG}" in
        fail_fast)
          fail_fast=true
          ;;
      esac
      ;;
    d)
      docker_image="${OPTARG}"
      ;;
    h)
      header_filter="${OPTARG}"
      ;;
    t)
      target_dir="${OPTARG}"
      ;;
    g)
      # OPTIND causes unbound variable error in get opt. It is caused by "set -u".
      # In order to suppress the error temporarily, call "set +u" here.
      set +u

      # NOTE:
      # https://stackoverflow.com/questions/7529856/retrieving-multiple-arguments-for-a-single-option-using-getopts-in-bash
      until [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [[ -z $(eval "echo \${$OPTIND}") ]]; do
        # shellcheck disable=SC2207
        git_diff_commits+=($(eval "echo \${$OPTIND}"))
        OPTIND=$((OPTIND + 1))
      done

      set -u
      ;;
    f)
      will_fix=true
      ;;
    c)
      cuda_arch="${OPTARG}"
      ;;
    p)
      dir_to_compile_commands_json="${OPTARG}"
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

if [ -z "${docker_image}" ]; then
  echo "docker image is required."
  usage
fi

IGNORE_WORDS=("3rdparty" "imgui" "unittest")

COMPILE_COMMANDS_JSON="${dir_to_compile_commands_json}/compile_commands.json"

# If compile_commands.json doesn't exist, generate.
if [ ! -e "${COMPILE_COMMANDS_JSON}" ]; then
  if [ -e "${THIS_DIR}/build.sh" ]; then
    ${THIS_DIR}/build.sh -d "${docker_image}" -e
  else
    echo "No script to build ${COMPILE_COMMANDS_JSON}"
    exit 1
  fi
fi

declare -a target_files=()

if ((${#git_diff_commits[*]} > 0)); then
  if ((${#git_diff_commits[*]} != 2)); then
    echo "2 commits need to be specified to -g option"
    exit 1
  fi
  # Get diff files by added or modified.
  # shellcheck disable=SC2207
  target_files=($(git diff --diff-filter=AM --name-only "${git_diff_commits[0]}" "${git_diff_commits[1]}" | grep -E ".*\.(cpp|cxx|h|hpp)$"))
fi

# In order to pass as arguments with option, insert option "-t" at top.
if ((${#target_files[*]} > 0)); then
  target_files=("-t" "${target_files[@]:0}")
elif [[ -d "${target_dir}" ]]; then
  readarray -d '' target_files < <(find "${target_dir}" -name *.cpp -type f -print0)
fi

options=()
if "${fail_fast}"; then
  options+=("--fail_fast")
fi

extra_args=()
if "${will_fix}"; then
  extra_args+=("--fix")
fi

# Specify compile_commands.json located directory.
# The element "directory" of the item is the same as the directory where cmake is run.
# And, the content of "directory" and the directory where compile_commands.json is located must be same.
# So, specify the directory where cmake is run to run clang-tidy.
extra_args+=("-p ${THIS_DIR}/../build")

# Not to use OpenMP because clang-tidy seems not to support it well.
extra_args+=("--extra-arg=-fno-openmp")

# Treat last element of docker image name as container name.

# shellcheck disable=SC2206
parsed_image_name=(${docker_image//\// })
# shellcheck disable=SC2206
parsed_image_name=(${parsed_image_name[-1]//:/ })
CONTAINER_NAME="${parsed_image_name[0]}"

# NOTE:
# It seems that python3 ./tools/run_clang_tidy.py via docker exec can't work in python subprocess.
# So, run docker exec separately here.
python3 ./tools/docker_operator.py -i "${docker_image}" -n "${CONTAINER_NAME}"

docker exec "${CONTAINER_NAME}" bash -c \
  "python3 ./tools/run_clang_tidy.py \
    -i ${IGNORE_WORDS[*]} \
    --header_filter ${header_filter} \
    -c ${COMPILE_COMMANDS_JSON} \
    -t ${target_files[*]} \
    ${options[*]} \
    -- ${extra_args[*]}"

#kill_container "${CONTAINER_NAME}"
