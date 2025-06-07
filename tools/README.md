<!-- markdownlint-disable MD033 -->
# Scripts

In this directory, there are some helper tools.

## build.sh

This is the helper script to run the docker container and execute the build command in the docker
container.

**Usage:**

```plain
Usage: build.sh [Options]
Options:
  -b <build_config>       : Build configuration. Default is "Release"
  -c <compute_capability> : Compute capability for CUDA. No need to specify ".". If it's "7.5", it's specified as "75". Default is "75"
  -d <docker_image>       : docker image to run build. This option is necessary
  -e                      : Only export compile_commands.json
```

e.g.

```bash
./tools/build.sh -b Release -c 75 -d ghcr.io/nackdai/aten/aten:latest
```

## <a name="clang_tidy_sh">clang_tidy.sh</a>

This is the helper script to run the docker container and run `clang-tidy` in the docker container.
This script actually runs the python script `run_clang_tidy.py`.

Option `-h` can specify header filter and it has the default value `src`. Option `-g` can specify
two git commits to specify only git diff files to clang-tidy. If `-g` is specified, `clang-tidy`
runs for only diff files between two commits.

**Usage:**

```plain
Usage: clang_tidy.sh [Options]
Options:
  -d <docker_image>    : docker image to run build. Required.
  -h <header_filter>   : Header filter. If nothing is specified, "src" is specified.
  -g <commit> <commit> : Specify git diff files as clang-tidy target files.
  -f                   : Fix code.
  --fail_fast          : If the error happens, stop immediately
```

e.g.

```bash
./tools/clang_tidy.sh -d ghcr.io/nackdai/aten/aten:latest -g -h ${PWD}/src --fail_fast
```

## docker_operator.py

This is the helper script to support the following docker operations:

* Run docker container and execute the command within the docker container
* Run docker container and enter docker container

**Usage:**

```plain
usage: docker_operator.py [-h] [-i IMAGE] [-n NAME] [-e] [-r] [-c COMMAND]

Run clang-tidy

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        docker image
  -n NAME, --name NAME  container name
  -e, --enter           Enter docker container
  -r, --remove          Remove docker container
  -c COMMAND, --command COMMAND
                        Commands to be executed
```

e.g.

```bash
python3 ./tools/docker_operator.py -i ghcr.io/nackdai/aten/aten_dev:latest -c "pre-commit run -a" -r
```

## enter_docker_container.sh

This is the helper script to run docker container and enter docker container. This script actually
runs the python script `docker_operator.py`.

**Usage:**

```plain
Usage: enter_docker_container.sh <docker_image>
  <docker_image>   : docker image to run build
```

e.g.

```bash
./tools/enter_docker_container.sh ghcr.io/nackdai/aten/aten:latest
```

## extract_file_from_vs_proj.py

This is the helper script to extract files which are specified in vs proj file. This script to help
updating CMakeFile. But, this script just extracting the files from vs proj file. The output is not
optimized to put CMakeFile directly. Therefore, after getting the outcome from the script, the
extracted files needs to be put to CMakeFile manually.

**Usage:**

```plain
usage: extract_file_from_vs_proj.py [-h] -v VCPROJ [-o OUTPUT] -b BASEPATH [-w WORKDIR]

Extract compile files from vs proj file

optional arguments:
  -h, --help            show this help message and exit
  -v VCPROJ, --vcproj VCPROJ
                        VC proj file to extract
  -o OUTPUT, --output OUTPUT
                        File to output
  -b BASEPATH, --basepath BASEPATH
                        Base path to convert to relative path
  -w WORKDIR, --workdir WORKDIR
                        Working directory
```

e.g.

```bash
python3 ./tools/extract_file_from_vs_proj.py -v vs2019/libaten.vcxproj -o libaten.txt -b src/libaten
```

## install_cuda.ps1

This is the helper script to install minimum necessary CUDA subpackages for aten. The purpose of
this script is installing CUDA subpackages in CI. This script is not considered using to establish
the local development environment. This is powershell script and this script works for only Windows.

**Usage:**

```plain
Usage: install_cuda.ps1 [Options]
  -no_download  : Flag if script doesn't download packages.
  -no_copy      : Flag if script doesn't copy packages.
```

e.g.

```batch
powershell -NoProfile -ExecutionPolicy Unrestricted .\tools\install_cuda.ps1
```

## run_clang_tidy.py

This is the python script to run `clang-tidy` actually. Basically, this script isn't used directly.
This is used by [clang_tidy.sh](#clang_tidy_sh)

**Usage:**

```plain
usage: run_clang_tidy.py [-h] [-i [IGNORE_WORDS [IGNORE_WORDS ...]]] [--header_filter HEADER_FILTER] [-t [TARGET_FILES [TARGET_FILES ...]]] [-f] [--fail_fast]

Run clang-tidy

optional arguments:
  -h, --help            show this help message and exit
  -i [IGNORE_WORDS [IGNORE_WORDS ...]], --ignore_words [IGNORE_WORDS [IGNORE_WORDS ...]]
                        List to ignore. If file path include specified work, that file is ignored
  --header_filter HEADER_FILTER
                        Header filter for clang-tidy
  -t [TARGET_FILES [TARGET_FILES ...]], --target_files [TARGET_FILES [TARGET_FILES ...]]
                        Specific target file to run clang-tidy
  -f, --fix             Fix code
  --fail_fast           If error happens, fails immediately
```

e.g.

```bash
python3 ./tools/run_clang_tidy.py -i 3rdparty imgui unittest --header_filter "${PWD}/src/" -t accelerator.cpp
```

## <a name="run_executable_sh">run_executable.sh</a>

This is the helper script to execute the specified executable in the docker container. This script
can pass the arguments to the specified executable directly.

**Usage:**

```plain
Usage: run_executable.sh [Options] -- <Args to executable>
Options:
  -b <directory>    : Base directory to store executables. This option is necessary
  -e <executable>   : Name to run executable. This option is necessary
  -d <docker_image> : docker image to run executable. This option is necessary

Args to executable: Arguments to pass to executable
```

e.g.

```bash
./tools/run_executable.sh -d build -e xxx -d ghcr.io/nackdai/aten/aten:latest -- -a
```

## run_unit_test.sh

This is the helper script to run aten's unit test in the docker container. This script just
executes [run_executable.sh](#run_executable_sh) internally. This script is specialized for running
aten's unit test. Therefore, the unit test executable is specified directly in this script.

**Usage:**

```plain
Usage: run_unit_test.sh <docker_image> <base_directory>
  <docker_image>   : docker image to run build
  <base_directory> : Base directory to store executables
```

e.g.:

```bash
./tools/run_unit_test.sh ghcr.io/nackdai/aten/aten:latest ./
```
