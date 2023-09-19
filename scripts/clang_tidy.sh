#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options]
Options:
  -d <docker_iamge>    : docker image to run build.
  -h <header_filter>   : Header filter. If nothing is specified, "aten/src" is specified.
  -g <commit> <commit> : Specify git diff files as clang-tidy target files.
ex) $0 -d aten_dev:latest -g -h ${PWD}/aten/src
EOF
  exit 1
}

docker_image=""
header_filter="${PWD}/aten/src"
declare -a git_diff_commits=()

while getopts "d:h:g" opt; do
  case "${opt}" in
    d)
      docker_image="${OPTARG}"
      ;;
    h)
      header_filter="${OPTARG}"
      ;;
    g)
      # OPTING causes unbound variable error in get opt. It is caused by "set -u".
      # In order to suprress the error temporarily, call "set +u" here.
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
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

IGNORE_WORDS=("3rdparty" "imgui" "unittest")

COMPILE_COMMANDS_JSON="compile_commands.json"

# If compile_commands.json doesn't exist, generate.
if [ ! -e "${COMPILE_COMMANDS_JSON}" ]; then
  this_script_path="$(dirname -- "${BASH_SOURCE[0]}")"
  if [ -e "${this_script_path}/build.sh" ]; then
    "${this_script_path}"/build.sh -d "${docker_image}" -e
  else
    echo "No script to build ${COMPILE_COMMANDS_JSON}"
    exit 1
  fi
fi

declare -a target_files=()

if ((${#git_diff_commits[*]} > 0)); then
  if ((${#git_diff_commits[*]} != 2)); then
    echo "2 commits need to be speified to -g option"
    exit 1
  fi
  # Get diff files by added or modified.
  # shellcheck disable=SC2207
  target_files=($(git diff --diff-filter=AM --name-only "${git_diff_commits[0]}" "${git_diff_commits[1]}" | grep -E ".*\.(cpp|cxx|h|hpp)$"))
fi

# In order to pass as arguments with option, insert option "-t" at top.
if ((${#target_files[*]} > 0)); then
  target_files=("-t" "${target_files[@]:0}")
fi

python3 ./scripts/docker_operator.py -r -i "${docker_image}" -c "python3 ./scripts/run_clang_tidy.py -i ${IGNORE_WORDS[*]} -f ${header_filter} ${target_files[*]}"
