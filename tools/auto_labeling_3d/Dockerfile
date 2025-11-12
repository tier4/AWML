ARG BASE_IMAGE=ghcr.io/tier4/awml:base-latest
FROM ${BASE_IMAGE}

# Install pip dependencies
RUN pip install python-json-logger filterpy==1.4.5

# Install tier4_perception_dataset
ARG perception_dataset_url=https://github.com/tier4/tier4_perception_dataset.git
ARG perception_dataset_version=v1.0.26

RUN git clone -b ${perception_dataset_version} ${perception_dataset_url} \
    && cd tier4_perception_dataset && pip install .

# Setup the supported models in default
WORKDIR /workspace
COPY projects projects

## Setup CenterPoint
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

## Setup BEVFusion
RUN python /workspace/projects/BEVFusion/setup.py develop \
    && pip install spconv-cu120
ENV PYTHONPATH=/workspace/projects
ENV PYTHONPATH=${PYTHONPATH}

## Setup StreamPETR
RUN pip install -e /workspace/projects/StreamPETR
