ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="2.1.0"
ARG MMENGINE="0.10.3"
ARG MMDET="3.2.0"
ARG MMDEPLOY="1.3.1"
ARG MMDET3D="1.4.0"
ARG MMPRETRAIN="1.2.0"
ARG MMSEGMENTATION="1.2.2"

ENV CUDA_HOME="/usr/local/cuda" \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.7 8.9+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install apt dependencies for base library
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

# Install pip dependencies for base library
RUN python3 -m pip --no-cache-dir install \
    aenum \
    gitpython \
    nptyping \
    numpy==1.23.5 \
    nvidia-pyindex \
    openmim \
    nltk==3.8.1

# Install mim components
RUN mim install \
    mmcv==${MMCV} \
    mmdeploy==${MMDEPLOY} \
    mmdet==${MMDET} \
    mmdet3d==${MMDET3D} \
    mmengine==${MMENGINE} \
    mmpretrain[multimodal]==${MMPRETRAIN} \
    mmsegmentation==${MMSEGMENTATION}

# Install rerun
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    libgtk-3-dev \
    libxkbcommon-x11-0
RUN python3 -m pip --no-cache-dir install \
    rerun-sdk==0.17.0

# Install t4-devkit
RUN python3 -m pip install git+https://github.com/tier4/t4-devkit@v0.0.7

# NOTE(knzo25): this patch is needed to use numpy versions over 1.23.5 (version used in mmdet3d 1.4.0)
# It can be safely deleted when mmdet3d updates the numpy version
COPY .patches/mmdet3d.patch /tmp/mmdet3d.patch
RUN cd $(python -c "import site; print(site.getsitepackages()[0])") \
  && git apply < /tmp/mmdet3d.patch \
  && rm -f /tmp/mmdet3d.patch \
  && cd /

ENV WGPU_BACKEND=gl

WORKDIR /workspace

COPY autoware_ml autoware_ml
COPY pipelines pipelines
COPY projects projects
COPY tools tools
COPY setup.py setup.py
COPY README.md README.md

RUN pip install --no-cache-dir -e .
