FROM nvidia/cudagl:11.2.2-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && apt-get install -y \
    build-essential \
    clang-8 \
    curl \
    gdb \
    git \
    libglu1-mesa-dev \
    lld-8 \
    lldb-8 \
    lsb-release \
    ninja-build \
    software-properties-common \
    wget \
    xorg-dev \
    xz-utils

# TODO
# CUDA 10.1 requires the clang version is less than 9.
#RUN curl -fL https://apt.llvm.org/llvm.sh | bash -s 11 \
#    && apt-get update \
#    && apt-get install -y clang-tidy-11

ARG cmake_version=3.21.3
RUN curl -fL https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-Linux-x86_64.sh > cmake.sh  \
    && sh ./cmake.sh --skip-license \
    && rm ./cmake.sh

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
