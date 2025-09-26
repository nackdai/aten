<!-- markdownlint-disable MD024 MD029 MD033 -->
# How to Build

## Preparation

We should also initialize submodules in the `3rdparty` directory.
To do this, run:

```bash
git submodule update --init --recursive
```

## Windows

1. Install `CUDA 12.5` and the required NVIDIA driver.
2. Run `aten/3rdparty/Build3rdParty.bat <BuildConfiguration> <VisualStudio edition>`
    - The first argument can be `Debug`, `Release`, etc.
    - The second argument can be `Community`, `Enterprise`, `BuildTools`, etc.
    - The defaults are `Release` and `Community`.
3. Open `aten/vs2019/aten.sln`.
4. Build the projects with the `x64` configuration (not `x86`).

The confirmed environment is `Visual Studio 2022` on Windows 10.

## Linux

1. Install `CUDA 12.5` and the required NVIDIA driver.
2. Install the necessary applications (see `env/aten/Dockerfile` for details):
    1. Install `cmake` version 3.21.3 or later.
    2. Install `clang` 12.
    3. Install `ninja-build`.
3. `cd aten/build`
4. `cp ../tools/RunCMake.sh ./`
5. `./RunCMake.sh <Build Type> <Compute Capability>`
6. Run `ninja`

The confirmed environment is Ubuntu 22.04.

### What is RunCMake.sh?

`RunCMake.sh` is a script to help we build `aten` with CMake.
It is located in the `tools` directory. If we want to use it, copy it to the build directory.

It requires two arguments:

1. Build Type: `Debug` or `Release`
2. Compute Capability: This depends on your GPU. Specify it without a dot. For example,
   if the Compute Capability is `7.5`, specify `75`.

Example usage:

```bash
./RunCMake.sh Release 75
```

We can get your Compute Capability by running `get_cuda_sm.sh`.
If we do not specify the Compute Capability, `get_cuda_sm.sh` will run during CMake configuration
and set it automatically.

## Docker (on Linux)

We can build and run the executables in a docker container.

1. Install Docker.
2. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).
3. Move to the `aten` directory.
4. Build the docker image:

    ```bash
    docker build -t <any_name> ./env/aten/
    ```

5. Run the docker container:

    ```bash
    docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <image_name>:latest bash
    ```

6. Inside the docker container, run:

    ```bash
    mkdir aten/build
    cd aten/build
    cp ../tools/RunCMake.sh .
    ./RunCMake.sh <Build Type> <Compute Capability>
    ninja
    ```

## docker-compose

1. Install Docker and docker-compose.
2. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).
3. Build the docker image:

    ```bash
    docker-compose -f .devcontainer/docker-compose.yml build
    ```

4. Run the docker container:

    ```bash
    docker-compose -f .devcontainer/docker-compose.yml run aten
    ```

5. Inside the docker container, run:

    ```bash
    mkdir aten/build
    cd aten/build
    cp ../tools/RunCMake.sh .
    ./RunCMake.sh <Build Type> <Compute Capability>
    ninja
    ```

## Pull pre-built docker image

If we need a pre-built docker image, we can pull it as follows:

```bash
docker pull ghcr.io/nackdai/aten/aten:latest
```

## Helper script

There is a script to help build libraries and executables.
This script fully depends on Docker and works only on Linux.

```bash
./tools/build.sh -b <build_config> -c <compute_capability> -d <docker_image>
```

For example:

```bash
./tools/build.sh -b Release -c 75 -d ghcr.io/nackdai/aten/aten:latest
```
