<!-- markdownlint-disable MD024 MD029 MD033 -->
# How to build

## Prepearation

You also should get submodules in `3rdparty` directory.
To do that, follow like the following:

```shell
git submodule update --init --recursive
```

## Windows

1. Install `CUDA 11.7` and depended NVIDIA driver
1. Run `aten/3rdparty/Build3rdParty.bat <BuildConfiguration> <VisualStudio edition>`
    - The first argument can accept `Debug`, `Release` etc.
    - The second argument can accept `Community`, `Enterprise`, `BuildTools` etc.
    - Default is `Release` and `Communitiy`
1. Launch `aten/vs2019/aten.sln`
1. Build the projects with `x64` (not support `x86`)

The confirmed environment is `Visual Studio 2019` on `Windows10`.

## Linux

1. Install `CUDA 11.7` and depended NVIDIA driver
1. Install applications (You can find what you need in `env/aten/Dockerfile`)
    1. Install `cmake` `3.21.3` or later
    1. Install `clang 12`
    1. Install `ninja-build`
1. `cd aten/build`
1. `cp ../scripts/RunCMake.sh ./`
1. `./RunCMake.sh <Build Type> <Compute Capability>`
1. Run make `ninja`

The confirmed environment is `Ubuntu 20.04`.

### What is RunCMake.sh

`RunCMake.sh` is a script to help you to build `aten` with CMake.
It is located in `scripts` directory. If you would like to use it.
Copy it to the build directory you want.

It needs 2 arguments like the followings:

1. Build Type: `Debug` or `Release`
1. Compute Capability: It depends on your GPU. But, you need to specify it
without `.`. For example, if `Compute Capability` is `7.5`, please specify
like `75`.

Example to run `RunCMake.sh` is the following:

```shell
./RunCMake.sh Release 75
```

You can get `Compute Capability` by running `get_cuda_sm.sh`.
If you don't specify `Compute Capability`, while configuring `CMakeLists.txt`,
`get_cuda_sm.sh` runs and `Compute Capability` is specified.

## Docker (on Linux)

You can build and run the executables in docker container.

1. Install `Docker`
2. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).
3. Move to `aten` directory
4. Build docker image like the following:

```shell
docker build -t <Any name> ./env/aten/
```

5. Run docker container like the following:

```shell
docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <Image name>:latest bash
```

6. In docker container, run the following commands:

```shell
mkdir aten/build
cd aten/build
cp ../scripts/RunCMake.sh .
./RunCMake.sh <Build Type> <Compute Capability>
ninja
```

## docker-compose

1. Install `Docker` and `docker-compose`
2. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).
3. Build docker image like the following:

```shell
docker-compose -f .devcontainer/docker-compose.yml build`
```

4. Run docker container like the following:

```shell
docker-compose .devcontainer/docker-compose.yml run aten
```

5. In docker container, run the following commands:

```shell
mkdir aten/build
cd aten/build
cp ../scripts/RunCMake.sh .
./RunCMake.sh <Build Type> <Compute Capability>
ninja
```

## Pull pre built docker image

If you need the pre-built docker image, you can pull the docker image like the following:

```shell
docker pull ghcr.io/nackdai/aten/aten:latest
```
## Helper script

There is a script to help building libraries and executables. The script fully depends on docker
and it works on only Linux.

```shell
./scripts/build.sh -b <build_config> -c <compute_capability -d <docker_iamge>
```

For example:

```shell
./scripts/build.sh -b Release -c 75 -d ghcr.io/nackdai/aten/aten:latest
```
