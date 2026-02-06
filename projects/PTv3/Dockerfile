ARG AWML_BASE_IMAGE="autoware-ml"
FROM ${AWML_BASE_IMAGE}

ENV FLASH_ATTN_CUDA_ARCHS="120"
ENV PYTHONPATH=/workspace/projects:/workspace/projects/PTv3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && . /root/.local/bin/env \
 && uv pip install --system --python /opt/conda/bin/python --upgrade pip setuptools wheel \
 && uv pip install --system --python /opt/conda/bin/python torch==2.8.0 torchvision==0.23.0 \
 && uv pip install --system --python /opt/conda/bin/python --no-build-isolation --no-build-isolation-package torch-cluster /workspace/projects/PTv3