# GHA (GitHub Action)

The Dockerfile in this directory creates an image which we can run github action locally.

And, `npm` is installed in the image, therefore we can develop JavaScript for
Github Action.

## How to realize

I use [act](https://github.com/nektos/act) to realize it.
And, `act` uses Docker in that process, therefore we also adopt
`Docker outside of Docker (DooD)`.

But, we can find `DooD` as one of `Docker in Docker` or alternative way of
`Docker in Docker` on the internet.

## What is DooD

Please refer [this](https://esakat.github.io/esakat-blog/posts/docker-in-docker/)
or [this](https://qiita.com/sugiyasu-qr/items/85a1bedb6458d4573407)

**Sorry both are Japanese.**

## How to build

We can build the image like below:

```shell
docker build -t <Image Name> .
```

## How to run

As I mentioned in [How-to-realize](#How-to-realize), the container depends on
`DooD`. We need to mount `docker.sock`, therfore we have to specify volume like
below:

```shell
docker run -it --rm -v /var/run/docker.sock:/var/run/docker.sock -v ${PWD}:/work gha:latest bash
```

According to [this](https://qiita.com/comefigo/items/6394a43b3bd97cde7b17)
(**sorry this also is Japanese**), we can mount same volume without minding
whether Windows or Linux.

About `act`, pleaser refer [act github repository](https://github.com/nektos/act)
