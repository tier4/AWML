ARG AWML_BASE_IMAGE="ghcr.io/tier4/awml:base-latest"
FROM ${AWML_BASE_IMAGE}
ARG TRT_VERSION=10.8.0.43

RUN python3 -m pip --no-cache-dir install tensorrt==${TRT_VERSION} pycuda
