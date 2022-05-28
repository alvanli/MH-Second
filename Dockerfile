# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 as base

USER root:root

WORKDIR /home/

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y --no-install-recommends 
RUN apt-get purge --auto-remove python3 && apt-get install -y python3.8
RUN apt-get install -y --force-yes software-properties-common lsb-release
RUN apt-get install -y git wget curl
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y python3-pip
RUN apt-get install -y libboost-all-dev
RUN apt-get update && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && apt-get install kitware-archive-keyring && rm /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt-get update && apt-get -y install cmake protobuf-compiler 

ADD OpenPCDet /home/OpenPCDet
# ADD spconv /home/spconv

RUN python3 -m pip --no-cache-dir install --upgrade pip setuptools && \
    python3 -m pip --no-cache-dir install numba==0.50.0 pillow==8.3.2 && \
    python3 -m pip --no-cache-dir install tensorboardX protobuf && \
    python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python3 -m pip --no-cache-dir install easydict spconv-cu111 numpy && \
    python3 -m pip install scipy PyYAML SharedArray scikit-image tqdm

WORKDIR "/home/OpenPCDet"
RUN python3 setup.py develop 

WORKDIR "/home/OpenPCDet/tools"
RUN python3 setup.py install

CMD ["bash"]