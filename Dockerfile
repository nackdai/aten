FROM nvidia/cudagl:10.1-devel-ubuntu16.04 

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y libglew-dev
RUN apt-get install -y gcc
RUN apt-get install -y g++
