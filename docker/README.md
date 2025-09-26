# Docker

This directory contains Docker-related files.

## nvidia cudagl

NVIDIA does not update their official `nvidia/cudagl` image very often. As a result, the official
image is often behind the latest CUDA version.
However, we can build it manually based on the [NVIDIA container-images/cuda repository](https://gitlab.com/nvidia/container-images/cuda),
which is included as a submodule at [3rdparty/nvidia_container_image_cuda](../3rdparty/nvidia_container_image_cuda).

We can find details on how to build the image at the [nvidia/cudagl Docker Hub](https://hub.docker.com/r/nvidia/cudagl),

For example:

```bash
./build.sh -d --image-name <image name> --cuda-version 12.5.1 --os ubuntu --os-version 22.04 --arch x86_64 --cudagl
```

The [build_docker_image.sh](./build_docker_image) script automates this process.

To get which CUDA version and which Ubuntu version are supported, visit [nvidia/cuda Docker hub](https://hub.docker.com/r/nvidia/cuda/tags)

## aten

The Dockerfile in this directory provides a minimal development environment.
The docker image built from the Dockerfile in the `aten` directory establishes the minimum
development environment and is based on the [nvidia/cudagl image](#nvidia-cudagl).
The image is typically named `aten`.

```bash
docker build -t ghcr.io/nackdai/aten/aten:latest --build-arg from=ghcr.io/nackdai/aten/nvidia/cudagl:12.5.1-devel-ubuntu22.04 -f docker/aten/Dockerfile docker
```

## dev

The Dockerfile in this directory provides an extended development environment.
The docker image built from the Dockerfile in the `dev` directory is intended to be based on `aten`
and includes additional linter and formatter tools. The image is typically named `aten_dev`.

```bash
docker build -t ghcr.io/nackdai/aten/aten_dev:latest --build-arg from=ghcr.io/nackdai/aten/aten:latest -f docker/dev/Dockerfile docker
```

## build_docker_image.sh

This is a helper script to build the docker images.

**Usage:**

```plain
Usage: build_docker_image.sh <OPTIONS>
  -b <build_context>         : Docker build context.
  -c <cuda_version>          : CUDA version. Default is 12.5.1
  -u <ubuntu_version>        : Ubuntu version. Default is 22.04
  -p <image_tag_prefix>      : Prefix for image tag
  --clear_cache              : Build with clearing cache.
```

Example:

```bash
build_docker_image.sh -b ./docker -p "ghcr.io/nackdai/aten" -c 12.5.1 -u 22.04
```

The default CUDA version is `12.5.1` and the default Ubuntu version is `22.04`.

## How to upgrade docker-compose

* Check where docker-compose is located:

```bash
which docker-compose
/usr/local/bin/docker-compose
```

* Remove docker-compose:

```bash
sudo rm -rf /usr/local/bin/docker-compose
```

* Download docker-compose:

We can place it anywhere as long as the location is in your PATH.

```bash
sudo curl -L https://github.com/docker/compose/releases/download/<version>/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

* Confirm that docker-compose works:

```bash
docker-compose version
```
