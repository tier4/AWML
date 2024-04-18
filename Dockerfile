ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="2.1.0"
ARG MMENGINE="0.10.3"
ARG MMDET="3.2.0"
ARG MMDEPLOY="1.3.1"
ARG MMDET3D="1.4.0"

ENV CUDA_HOME="/usr/local/cuda" \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.7 8.9+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install apt dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
RUN python3 -m pip --no-cache-dir install \
    aenum \
    nptyping \
    numpy==1.23.5 \
    nvidia-pyindex \
    openmim

# Install mim components
RUN mim install \
    mmcv==${MMCV} \
    mmdeploy==${MMDEPLOY} \
    mmdet==${MMDET} \
    mmdet3d==${MMDET3D} \
    mmengine==${MMENGINE}

WORKDIR /workspace

COPY autoware_ml autoware_ml
COPY configs configs
COPY projects projects
COPY tools tools
COPY setup.py setup.py
COPY README.md README.md

RUN pip install --no-cache-dir -e .
