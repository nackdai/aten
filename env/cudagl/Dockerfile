# NOTE:
# If the newer version is necessary, we need to create docker image manually with the following way.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cudagl

ARG from

FROM ${from}

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
