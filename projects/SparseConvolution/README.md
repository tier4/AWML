# SparseConvolutions
## Summary

This is essentially a copy of MMCV's sparse convolution implementation.
It is known not to be the fastest, but sufficies to use it at runtime for model export

## Get started
### Export as ONNX

We provide a simple example of ONXX export in `example_export_onnx.py`.
To use it simply run `python example_export_onnx.py`

### TensorRT inference

Only 2D sparse convolutions have been tested as a separate module.
More complex architectures like the ones in PTv3 and BEVFusion are cooming soon

## Troubleshooting

## Reference

- [mmcv/mmcv/ops](https://github.com/open-mmlab/mmcv/tree/main/mmcv/ops)
