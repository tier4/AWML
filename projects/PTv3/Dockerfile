ARG AWML_BASE_IMAGE="autoware-ml"
FROM ${AWML_BASE_IMAGE}

RUN python3 -m pip --no-cache-dir install \
    SharedArray \
    open3d \
    spconv-cu120 \
    flash-attn \
    yapf==0.40.1 \
    torch-geometric \
    ftfy \
    regex \
    tqdm

RUN python3 -m pip --no-cache-dir install \
  git+https://github.com/openai/CLIP.git

RUN conda install h5py pyyaml -c anaconda -y
RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

RUN python3 -m pip --no-cache-dir install --force-reinstall \
    yapf==0.40.1

ENV PYTHONPATH=/workspace/projects:/workspace/projects/PTv3

# For Blackwell, as of 2025/05, some packages need to be compiled or use nightly versions
# RUN python3 -m pip --no-cache-dir install git+https://github.com/rusty1s/pytorch_scatter.git@2.1.2 -vvv
# RUN python3 -m pip --no-cache-dir install torch-geometric -f https://data.pyg.org/whl/nightly/torch-2.7.0%2Bcu128.html
# ENV FLASH_ATTN_CUDA_ARCHS="120"
# MAX_JOBS=4 python3 -m pip --no-cache-dir install flash-attn --no-build-isolation -vvv
