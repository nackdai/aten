# Docker

This directory contains docker related files.

## nvidia cudagl

NVIDIA basically doesn't update their official nvidia/cudagl image so often. Thus, unfortunately,
NVIDIA official nvidia/cudagl docker image is behind from CUDA version. But, we can build it
manually based on [NVIDIA container-images/cuda repository](https://gitlab.com/nvidia/container-images/cuda).
It's added as submodule at [3rdparty/nvidia_container_image_cuda](../3rdparty/nvidia_container_image_cuda).

We can find the detail of the command to build it at [nvidia/cudagl docker hub](https://hub.docker.com/r/nvidia/cudagl),
like the following:

```sh
./build.sh -d --image-name <image name> --cuda-version 11.7.1 --os ubuntu --os-version 20.04 --arch x86_64 --cudagl
```

[build_docker_image.sh](./build_docker_image) does it in its script.

## aten

Dockerfile in this directory is for the minimum development environment. The docker image by
Dockerfile in aten directory establish the minimum development environment. The docker imgae is
based on the [nvidia/cudagl image](#nvidia-cudagl). The docker image is basically named as
`aten`.

```shell
docker build -t ghcr.io/nackdai/aten/aten:latest --build-arg from=ghcr.io/nackdai/aten/nvidia/cudagl:11.7.1-devel-ubuntu20.04 -f docker/aten/Dockerfile docker
```

## dev

Dockerfile in this directory is for the extra development environment The docker image by
Dockerfile in dev directory is intended to be based on `aten`. It extends to isntall linter and
formatter tools. The docker image is basically named as `aten_dev`

```shell
docker build -t ghcr.io/nackdai/aten/aten_dev:latest --build-arg from=ghcr.io/nackdai/aten/aten:latest -f docker/aten/Dockerfile docker
```

## build_docker_image.sh

This the helper scrpt to build the docker images.

**Usage:**

```plain
Usage: build_docker_image.sh <OPTIONS>
  -b <build_context>         : Docker build context.
  -n <nvidia_cuda_image_tag> : Base nvidia/cuda docker image
  -p <image_tag_prefix>      : Prefix for image tag
```

e.g.

```shell
build_docker_image.sh -b docker -n 11.7.1-devel-ubuntu20.04 -p ghcr.io/nackdai/aten
```

If `-n` is not specified, the default is `11.7.1-devel-ubuntu20.04`.

## How to upgrade docker-compose

* Check where docker-compose is located:

```sh
$ which docker-compose
/usr/local/bin/docker-compose
```

* Remove docker-compose

```sh
sudo rm -rf /usr/bin/docker-compose
```

* Download docker-cmpose

We can locate where ever we want, if that location is in PATH.

```sh
sudo curl -L https://github.com/docker/compose/releases/download/<version>/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

* Confirm if docker-compose works

```sh
docker-compose version
```
