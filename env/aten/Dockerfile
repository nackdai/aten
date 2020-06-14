FROM nvidia/cudagl:10.1-devel-ubuntu18.04

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
    ninja-build \
    wget \
    xorg-dev \
    xz-utils

RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2-Linux-x86_64.sh \
    && sh ./cmake-3.15.2-Linux-x86_64.sh --skip-license \
    && rm ./cmake-3.15.2-Linux-x86_64.sh

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# For 16.04

# cmake
# https://www.osetc.com/en/how-to-install-the-latest-version-of-cmake-on-ubuntu-16-04-18-04-linux.html
#RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0-Linux-x86_64.tar.gz \
#    && tar -xvf cmake-3.15.0-Linux-x86_64.tar.gz \
#    && rm cmake-3.15.0-Linux-x86_64.tar.gz \
#    && mv cmake-3.15.0-Linux-x86_64 /opt \
#    && ln -s /opt/cmake-3.15.0-Linux-x86_64/bin/* /usr/bin

# clang
# https://solarianprogrammer.com/2017/12/13/linux-wsl-install-clang-libcpp-compile-cpp-17-programs/
#RUN curl -SL http://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz | tar -xJC .\
#    && mv clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-16.04 /usr/local/clang_8.0.0
#ENV PATH /usr/local/clang_8.0.0/bin:${PATH}

#RUN apt-get install -y libglfw3-dev
#RUN apt-get install -y libglew-dev
