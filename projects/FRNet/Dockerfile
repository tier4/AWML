FROM autoware-ml

ARG TRT_MAJOR=10.4.0

# Install pip dependencies
RUN python3 -m pip --no-cache-dir install \
    onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    onnxscript \
    onnx_graphsurgeon \
    polygraphy \
    pycuda \
    tensorrt==${TRT_MAJOR} \
    tensorrt-cu12==${TRT_MAJOR}

WORKDIR /workspace
RUN pip install --no-cache-dir -e .
