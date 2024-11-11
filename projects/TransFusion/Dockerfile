ARG AWML_BASE_IMAGE="ghcr.io/tier4/autoware-ml-base:latest"
FROM ${AWML_BASE_IMAGE}

# Install pip dependencies (1)
RUN python3 -m pip --no-cache-dir install \
    tensorrt

# Install pip dependencies (depend on 1)
RUN python3 -m pip --no-cache-dir install \
    onnx-graphsurgeon

# Apply patches
COPY mmdeploy.patch /tmp/mmdeploy.patch
RUN cd $(python -c "import site; print(site.getsitepackages()[0])") \
  && git apply < /tmp/mmdeploy.patch \
  && rm -f /tmp/mmdeploy.patch \
  && cd /

WORKDIR /workspace
RUN pip install --no-cache-dir -e .
