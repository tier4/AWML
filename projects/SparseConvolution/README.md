# SparseConvolutions

This is an auxiliar project that enables ONNX export of sparse convolutions.
Currently, we only support traveller59's backend

# ONNX Export

To enable onnx export for sparse convolutions, this project must be imported in the target model.
for example `import projects.SparseConvolutions` will overwrite the sparse convolution symbols for some that are suited for exporting.

The recommended way to do this is using mmcv's way using custom imports:

```python
custom_imports = dict(
    imports=[
        'projects.SparseConvolution',
    ],
    allow_failed_imports=False)
```

An example is presented in [BEVFusion](../BEVFusion/README.md)

# TensorRT inference

An example of sparse convolution inference in TensorRT is presented in our BEVFusion [implementation](https://github.com/knzo25/bevfusion_ros2).
