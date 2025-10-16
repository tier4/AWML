ARG AWML_BASE_IMAGE="ghcr.io/tier4/autoware-ml-base:latest"
FROM ${AWML_BASE_IMAGE}

# Install SparseConv
RUN python3 -m pip --no-cache-dir install \
  spconv-cu120

# Install the libraries for onnx deployment
RUN python3 -m pip --no-cache-dir install \
  onnx==1.17.0 \
  onnx_simplifier==0.4.8 \
  onnxsim==0.4.10 \
  onnxruntime==1.13.1 \
  shapely==1.8.0

RUN python3 -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.ngc.nvidia.com \
  --extra-index-url https://pypi.nvidia.com \
  onnx-graphsurgeon \
  pytorch-quantization

RUN python3 -m pip --no-cache-dir install \
  numpy==1.22

WORKDIR /workspace
RUN pip install --no-cache-dir -e .
