<!-- markdownlint-disable MD024 MD029 MD033 -->
# How to Run

## Windows

We can find and run the `.exe` files in each directory where the source files are located.

## Linux

We can find and run the executables in the directories where we built the applications.
The directories have the same names as the executable files.

## <a name="RunOnDocker">Docker</a>

This section applies only to Linux.

If we would like to run the executables in the docker container,
we need to ensure that your host can accept X11 forwarded connections:

```bash
xhost +local:<docker container name>
```

Then, run the docker container as follows:

```bash
docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <image_name>:latest bash
```

### docker-compose

We also need to ensure your host accepts X11 forwarded connections.
See [Docker in How to Run](#RunOnDocker).

Then, run the docker container via docker-compose as follows:

```bash
docker-compose -f .devcontainer/docker-compose.yml run aten
```
