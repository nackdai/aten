ARG from=ghcr.io/nackdai/aten/nvidia/cudagl:12.5.1-devel-ubuntu22.04
FROM ${from}

ARG ubuntu_codename="jammy"

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
    apt-utils \
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

# NOTE:
# clang newer than 14 requires to install libomp-${clang_version}-dev.
# https://github.com/llvm/llvm-project/issues/56460

# NOTE:
# clang-13 breaks OpenMP, "omp_is_initial_device" has already been defined.
# https://github.com/alpaka-group/alpaka/issues/1755

# https://apt.llvm.org/
# NOTE: When we specify 'all' option, installing some of packages is failed.

# NOTE:
# llvm.sh seems not to be reliable...
# https://github.com/llvm/llvm-project/issues/54676
ARG clang_version=17
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && add-apt-repository -y "deb http://apt.llvm.org/${ubuntu_codename}/ llvm-toolchain-${ubuntu_codename}-${clang_version} main" \
    && apt-get update -y \
    && apt-get install -y \
    libllvm${clang_version} \
    llvm-${clang_version}-dev \
    libclang-cpp${clang_version} \
    libclang-common-${clang_version}-dev \
    clang-${clang_version} \
    lldb-${clang_version} \
    lld-${clang_version} \
    clangd-${clang_version} \
    clang-tidy-${clang_version} \
    libc++-${clang_version}-dev \
    libc++abi-${clang_version}-dev \
    libomp-${clang_version}-dev \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/clang-${clang_version} /usr/local/bin/clang \
    && ln -s /usr/bin/clangd-${clang_version} /usr/local/bin/clangd \
    && ln -s /usr/bin/clang-tidy-${clang_version} /usr/local/bin/clang-tidy \
    && ln -s /usr/bin/lldb-${clang_version} /usr/local/bin/lldb

ARG cmake_version=3.21.3
RUN curl -fL https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}-Linux-x86_64.sh > cmake.sh  \
    && mkdir -p /bin/cmake \
    && sh ./cmake.sh --skip-license --prefix=/bin/cmake \
    && rm ./cmake.sh
ENV PATH $PATH:/bin/cmake/bin
