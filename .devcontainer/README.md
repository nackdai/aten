# devcontainer

## Build docker image

docker-compose doesn't support build order. It means we can't build the docker images which has the
dependency. For example, we can't build the base image and then build the another image based on
the base image with docker-compose. As aten, `aten_dev` docker image requires the base image
`aten`.

Therefore, we need to build docker image beforehand for devcontainer with the following command.

```sh
./docker/build_docker_image.sh -b ./docker -p aten
```

And then, `docker.io/aten/aten_dev:latest` can be created. That docker image is already specified
in `.devcontainer/docker-compose.yml`, and it is specified in `.devcontainer/devcontainer.json`
as well.

## Launch devcontainer

We can launch devcontainer as usual in vscode.

### GPU available case

In `devcontainer.json`, `initializeCommand` line is commnted out. Because, it doesn't work the the
enviroment whici there is no GPU. If we'd like to launch the executable to require the window, GPU
within devcontainer, `initializeCommand` has to be valid.
