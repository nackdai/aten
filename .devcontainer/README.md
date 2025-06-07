# devcontainer

## Build docker image

docker-compose does not support build order. This means we cannot build docker images with
dependencies in the correct order using docker-compose alone.
For example, we cannot build a base image and then build another image based on that base image
with a single docker-compose command.
In aten, the `aten_dev` docker image requires the base image `aten`.

Therefore, we need to build the docker image beforehand for the devcontainer with the following command:

```bash
./docker/build_docker_image.sh -b ./docker -p aten
```

After that, `docker.io/aten/aten_dev:latest` will be created.
This docker image is already specified in `.devcontainer/docker-compose.yml`,
and it is also referenced in `.devcontainer/devcontainer.json`.

## Launch devcontainer

We can launch the devcontainer as usual in VSCode.

### When GPU is available

In `devcontainer.json`, the `initializeCommand` line is commented out because it does not work in
environments without a GPU.
If we want to launch executables that require a window or GPU within the devcontainer,
we need to uncomment and enable the `initializeCommand`.
