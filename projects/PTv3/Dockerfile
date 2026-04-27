ARG AWML_BASE_IMAGE="ghcr.io/tier4/awml:base-latest"
FROM ${AWML_BASE_IMAGE}

ENV FLASH_ATTN_CUDA_ARCHS="120"
ENV PYTHONPATH=/workspace/projects:/workspace/projects/PTv3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && . /root/.local/bin/env \
 && uv pip install --system --python /opt/conda/bin/python \
    --upgrade pip setuptools wheel \
 && uv pip install --system --python /opt/conda/bin/python \
    --no-build-isolation flash-attn /workspace/projects/PTv3
