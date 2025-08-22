ARG AWML_BASE_IMAGE="autoware-ml:latest"
FROM ${AWML_BASE_IMAGE}
ARG TRT_VERSION=10.8.0.43

# Install pip dependencies
RUN python3 -m pip --no-cache-dir install \
  onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxsim \
  pycuda \
  tensorrt-cu12==${TRT_VERSION} \
  "transformers<=4.37.2" \
  transforms3d

WORKDIR /workspace
RUN pip install --no-cache-dir -e .
