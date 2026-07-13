ARG AWML_BASE_IMAGE="autoware-ml:latest"
FROM ${AWML_BASE_IMAGE}

ARG TRT_VERSION=10.8.0.43
ARG BUILD_SPCONV_CPP="true"
ARG SPCONV_CPP_REF="main"
ARG BUILD_TRT_PLUGINS="true"
# AWML fork of autoware.universe that carries the GetIndicePairsImplicitGemm
# `do_sort` plugin attribute change required by the BEVFusion sparse ONNX export
# (stock upstream autowarefoundation main lacks this attribute).
ARG AUTOWARE_UNIVERSE_REPO="https://github.com/vividf/autoware.universe.git"
ARG AUTOWARE_UNIVERSE_REF="feat/implicit_gemm_int8"

# Install deployment/runtime dependencies
RUN python3 -m pip --no-cache-dir install \
  onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxsim \
  pycuda \
  tensorrt-cu12==${TRT_VERSION}

# Install traveller59 python package (requested)
RUN python3 -m pip --no-cache-dir install spconv-cu120

# Build and install libspconv.so from spconv_cpp (for C++/TensorRT deployment path)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  wget && \
  rm -rf /var/lib/apt/lists/*

RUN if [ "${BUILD_SPCONV_CPP}" = "true" ]; then \
    git clone --depth 1 --branch "${SPCONV_CPP_REF}" https://github.com/autowarefoundation/spconv_cpp.git /opt/spconv_cpp && \
    cd /opt/spconv_cpp && \
    mkdir -p cumm/build-amd64 && \
    cd cumm/build-amd64 && \
    cmake .. && make -j"$(nproc)" && cpack -G DEB && \
    apt-get update && apt-get install -y /opt/spconv_cpp/cumm/_packages/cumm_0.5.3_amd64.deb && \
    cd /opt/spconv_cpp && \
    mkdir -p spconv/build-amd64 && \
    cd spconv/build-amd64 && \
    cmake .. && make -j"$(nproc)" && cpack -G DEB && \
    apt-get update && apt-get install -y /opt/spconv_cpp/spconv/_packages/spconv_2.3.8_amd64.deb && \
    ldconfig; \
  fi

WORKDIR /workspace
RUN python3 projects/BEVFusion/setup.py develop

# Build TensorRT custom plugins at image build time.
COPY projects/BEVFusion/plugins/build_plugin_inside_container.sh /workspace/projects/BEVFusion/plugins/build_plugin_inside_container.sh
COPY projects/BEVFusion/plugins/CMakeLists.standalone /workspace/projects/BEVFusion/plugins/CMakeLists.standalone
RUN if [ "${BUILD_TRT_PLUGINS}" = "true" ]; then \
    chmod +x /workspace/projects/BEVFusion/plugins/build_plugin_inside_container.sh && \
    AUTOWARE_UNIVERSE_REPO="${AUTOWARE_UNIVERSE_REPO}" \
    AUTOWARE_UNIVERSE_REF="${AUTOWARE_UNIVERSE_REF}" \
    INSTALL_PLUGINS_DIR="/opt/plugins" \
    bash /workspace/projects/BEVFusion/plugins/build_plugin_inside_container.sh; \
  fi
