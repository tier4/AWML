FROM autoware-ml

# Install pip dependencies
RUN python3 -m pip --no-cache-dir install \
    onnxruntime==1.18.0 \
    onnxscript \
    onnx_graphsurgeon \
    tensorrt==8.6.1

# Apply patches
COPY torch_utils.patch /tmp/torch_utils.patch
RUN cd $(python -c "import site; print(site.getsitepackages()[0])") \
  && git apply < /tmp/torch_utils.patch \
  && rm -f /tmp/torch_utils.patch \
  && cd /

WORKDIR /workspace
RUN pip install --no-cache-dir -e .
