<!-- markdownlint-disable MD024 MD029 MD033 -->
# How to run

## Windows

We can find `exe` files and run them. You can find them in each directory
where the source files are in.

## Linux

We can find the executables and run them. You can find them in the directories which you built the
applications. And the directories have same name as execution file.

## <a name="RunOnDocker">Docker</a>

This section works for ony Linux.

If we would like to run the executables in docker, we need to ensure that your host can accept
X forwarded connections:

```bash
xhost +local:<Docker container name>`
```

And then, run the docker container like the following:

```bash
docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <Image Name>:latest bash
```

### docker-compose

We also need to ensure your host accept X forward connections.
See [Docker in How to run](#RunOnDocker)

And, run the docker container via docker-compose like the following:

```bash
docker-compose -f .devcontainer/docker-compose.yml run aten
```
