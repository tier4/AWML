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
RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

RUN python3 -m pip --no-cache-dir --force-reinstall install \ 
    yapf==0.40.1
