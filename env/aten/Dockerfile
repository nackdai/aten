ARG from=nvidia/cudagl:11.4.2-devel-ubuntu20.04
FROM ${from}

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gdb \
    git \
    libglu1-mesa-dev \
    lsb-release \
    ninja-build \
    software-properties-common \
    wget \
    xorg-dev \
    xz-utils \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# NOTE: It seems that there are some bugs for combination cuda and clang. In order to be stable, we install clang-9.
# https://forums.developer.nvidia.com/t/cuda-is-not-compatible-with-current-versions-of-libc/190930
# https://forums.developer.nvidia.com/t/building-with-clang-cuda-11-3-0-works-but-with-cuda-11-3-1-fails-regression/182176/3

# https://apt.llvm.org/
# NOTE: When we specify 'all' option, installing some of packages is failed.
ARG clang_version=9
RUN curl -fL https://apt.llvm.org/llvm.sh | bash -s ${clang_version} \
    && apt-get update \
    && apt-get install -y clang-tidy-${clang_version} libc++-${clang_version}-dev libc++abi-${clang_version}-dev \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG cmake_version=3.21.3
RUN curl -fL https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-Linux-x86_64.sh > cmake.sh  \
    && mkdir -p /bin/cmake \
    && sh ./cmake.sh --skip-license --prefix=/bin/cmake \
    && rm ./cmake.sh
ENV PATH $PATH:/bin/cmake/bin
