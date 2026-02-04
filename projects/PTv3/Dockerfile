ARG AWML_BASE_IMAGE="autoware-ml"
FROM ${AWML_BASE_IMAGE}

ENV FLASH_ATTN_CUDA_ARCHS="120"
ENV PYTHONPATH=/workspace/projects:/workspace/projects/PTv3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && . /root/.local/bin/env \
 && echo UV_PROJECT_ENVIRONMENT="/opt/conda/" \
 && uv sync --project /workspace/projects/PTv3