#!/bin/bash

set -eu
set -o pipefail

usage() {
  cat <<EOF 1>&2
Usage: $0 [Options]
Options:
  -b <build_config>       : Build configuration. Default is "Release"
  -c <compute_capability> : Compute capability for CUDA. No need to specify ".". If it's "7.5", it's specified as "75". Default is "75"
  -d <docker_iamge>       : docker image to run build. This option is necessary
  -e                      : Only export compile_commands.json
ex) $0 -b Release -c 75 -d aten:latest
EOF
  exit 1
}

build_type="Release"
compute_capability="75"
docker_image=""
will_export_compile_commands_json=false

while getopts "b:c:d:e" opt; do
  case "${opt}" in
    b)
      build_type="${OPTARG}"
      ;;
    c)
      compute_capability="${OPTARG}"
      ;;
    d)
      docker_image="${OPTARG}"
      ;;
    e)
      will_export_compile_commands_json=true
      ;;
    \?)
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

echo "Build config: [${build_type}]"
echo "Compute Capability: [${compute_capability}]"

cmake_cmd="cmake \
  -D CMAKE_BUILD_TYPE=${build_type} \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-12 \
  -D CMAKE_C_COMPILER=/usr/bin/clang-12 \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -D CUDA_HOST_COMPILER=/usr/bin/clang-12 \
  -D CUDA_TARGET_COMPUTE_CAPABILITY=${compute_capability} \
  -G Ninja .."

if "${will_export_compile_commands_json}"; then
  # Just generating compile_commands_json. So, docker container should be removed.
  python3 ./scripts/docker_operator.py -r -i "${docker_image}" -c "mkdir -p build && cd build && ${cmake_cmd}"
else
  # In first running, not remove docker container for second running.
  python3 ./scripts/docker_operator.py -i "${docker_image}" -c "mkdir -p build && cd build && ${cmake_cmd}"
fi

COMPILE_COMMANDS_JSON="compile_commands.json"

# Just exporting compile_commands.json.
if "${will_export_compile_commands_json}"; then
  if [ -e "${PWD}/build/${COMPILE_COMMANDS_JSON}" ]; then
    mv "${PWD}/build/${COMPILE_COMMANDS_JSON}" "${PWD}/${COMPILE_COMMANDS_JSON}"
    exit 0
  else
    echo "Not generated ${COMPILE_COMMANDS_JSON}"
    exit 1
  fi
fi

# No need to keep docker conainer after running command. So, specify remove option.
python3 ./scripts/docker_operator.py -r -i "${docker_image}" -c "cd build && ninja -j 4"
