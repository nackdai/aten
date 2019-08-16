FROM nvidia/cudagl:10.1-devel-ubuntu16.04 

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update

# cmake
# https://www.osetc.com/en/how-to-install-the-latest-version-of-cmake-on-ubuntu-16-04-18-04-linux.html
RUN apt-get install -y wget
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0-Linux-x86_64.tar.gz
RUN tar -xvf cmake-3.15.0-Linux-x86_64.tar.gz
RUN rm cmake-3.15.0-Linux-x86_64.tar.gz
RUN mv cmake-3.15.0-Linux-x86_64 /opt
RUN ln -s /opt/cmake-3.15.0-Linux-x86_64/bin/* /usr/bin

# clang
# https://solarianprogrammer.com/2017/12/13/linux-wsl-install-clang-libcpp-compile-cpp-17-programs/
RUN apt-get install -y build-essential xz-utils curl
RUN curl -SL http://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz | tar -xJC .
RUN mv clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-16.04 /usr/local/clang_8.0.0
ENV PATH /usr/local/clang_8.0.0/bin:${PATH}

#RUN apt-get install -y libglfw3-dev
#RUN apt-get install -y libglew-dev
RUN apt-get install -y xorg-dev
RUN apt-get install -y libglu1-mesa-dev

RUN apt-get install -y gcc
RUN apt-get install -y g++

RUN apt-get install -y ninja-build
