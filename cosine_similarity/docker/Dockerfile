FROM nvidia/cuda:11.2.1-devel-ubuntu20.04
#FROM osrf/ros:melodic-desktop-full

# author
MAINTAINER Ray Zhang

LABEL version="0.5"
LABEL description="Docker file: for setting up CVO-GPU"

# optional, if the default user is not "root", you might need to switch to root here and at the end of the script to the original user again.
# e.g.
# USER root
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8


RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        dialog \
        apt-utils \
        libeigen3-dev \
        git \
        mercurial \
        pkg-config \
        mesa-utils \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev \ 
        git \
        tmux \
        wget \
        curl \
        cmake \
        vim \ 
        build-essential \
        unzip \
        autoconf \
        autogen \
        automake \
        libtool \
        mlocate \
        zlib1g-dev \
        libopenblas-dev \
        qt5-default \
        libflann-dev \
        libopencv-dev \
        libpcl-dev \
        libboost-all-dev \
        libsuitesparse-dev \
        libtbb-dev \
        libusb-1.0-0-dev \
        libgtest-dev \
        pkg-config \
        cmake-curses-gui \
        emacs && \
    apt clean  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

######## Install Eigen 3.3.7
RUN  mkdir -p /home/root/tmp/ \
&&   wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 \
&&   tar -xf eigen-3.3.7.tar.bz2 && rm eigen-3.3.7.tar.bz2 \
&&   mv eigen-3.3.7 /home/root/tmp/eigen-3.3.7 \
&&   cd /home/root/tmp/eigen-3.3.7  \
&&   mkdir build && cd build && cmake .. && make -j && make install 
#-DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT && make -j && make install

######## Install sophus
RUN  mkdir -p /home/root/tmp/ \
&& cd /home/root/tmp && git clone https://github.com/strasdat/Sophus.git \
&& cd /home/root/tmp/Sophus \
&& mkdir build && cd build && cmake .. && make -j && make install

########### Install gtsam
RUN cd /home/root/tmp/ \
&& git clone https://github.com/borglab/gtsam.git \
&& cd gtsam \
&& mkdir build && cd build && cmake .. && make -j && make install

########### Install pcl-1.9.1
RUN cd /home/root/tmp/ \
&& wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.1.tar.gz \
&& tar -xf pcl-1.9.1.tar.gz && rm pcl-1.9.1.tar.gz \
&& mv pcl-pcl-1.9.1 pcl-1.9.1  \
&& mkdir -p pcl-1.9.1/build

COPY  boost.h  /home/root/tmp/pcl-1.9.1/io/include/pcl/io/boost.h

RUN cd /home/root/tmp/pcl-1.9.1/build && cmake .. && make -j && make install

RUN rm -rf /home/root/tmp/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# USER nvidia
