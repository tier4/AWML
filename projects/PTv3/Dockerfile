ARG AWML_BASE_IMAGE="autoware-ml"
FROM ${AWML_BASE_IMAGE}

ENV FLASH_ATTN_CUDA_ARCHS="120"
ENV PYTHONPATH=/workspace/projects:/workspace/projects/PTv3

RUN python3 -m pip --no-cache-dir install \
  addict \
  open3d \
  flash-attn --no-build-isolation \
  regex \
  sharedarray \
  spconv-cu120 \
  tensorboardx \
  termcolor \
  torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html \
  tqdm \
  yapf==0.40.1
