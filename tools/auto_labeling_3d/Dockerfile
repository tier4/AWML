ARG BASE_IMAGE=ghcr.io/tier4/autoware-ml-base:latest
FROM ${BASE_IMAGE}

# Install pip dependencies
RUN pip install python-json-logger filterpy==1.4.5

# Install tier4_perception_dataset
ARG perception_dataset_url=https://github.com/tier4/tier4_perception_dataset.git
ARG perception_dataset_version=v1.0.26

RUN git clone -b ${perception_dataset_version} ${perception_dataset_url} \
    && cd tier4_perception_dataset && pip install .
