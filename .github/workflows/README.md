# Workflows

## CI

In CI worflow (ci.yml), there are 2 jobs:

1. build_docker
1. build

### build_docker

literally, this job creates Docker image. But, usually creating docker image
take a long time. Therfore, I would like to avoid it as much as possible. To
do that, before creating Docker image, whether Dockerfile is changed is
checked. Flow of this job is below:

1. Checkout
To run local Github Actions JavaScript, need to checkout repository.
2. Get diff files
To check Dockerfiles is changed, check changed files in commits.
3. Dump
Display outcome of `Get diff files`
4. Build and push image
If `.devcontainer/Dockerfile` is in changed file lists from `Get diff files`
step, this step will be run.

    * Build Docker image
    * To push created Docker image to Github retistry, login Github retistry
    * Put tag
    * Push the image

We can separate this job to another workflow. But, the Docker image have
building envrironment. Therefore, I would like to use created image for next
job. To do that, this job have to run before `build` job.

### build

This job builds this repository. This job is run in the Docker image which is
stored in Github registry. Unfortunately, it seems that there is no way to use
the Docker image which is stored in Github registry with `container` workflow
syntax. Flow of this jbo is below:

1. Checkout
To build this repositroy, need to checkout not only repository but also
submodules.
2. Login registry
As I wrote, 8nfortunately, it seems that there is no way to use
the Docker image which is stored in Github registry with `container` workflow
syntax. Therefore, to pull the Docker image, need to login Github registry.
3. Pull image
Pull image from the login-ed Github registry
4. Run image
Run the pulled image. To execute command with `docker exec`, need to specify
container name with `--name` option.
5. Configure
Unfortunately, we can't specify pulled image with `uses` worflow syntax.
Therefore, we have to execute command with `docker exec`. This steps run
`CMake`.
6. Build
Run `ninja` to build.
