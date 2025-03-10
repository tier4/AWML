# Performance Measurement Tools

This directory contains tools for measuring and comparing the performance of ML models across different frameworks (PyTorch, TensorRT).


## Tools Overview

The repository contains three main tools:

1. `torch_time_measure.py`: Measures inference time for PyTorch models
2. `onnx_to_tensorrt.py`: Converts ONNX models to TensorRT engines
3. `tensorrt_time_measure.py`: Measures inference time for TensorRT engines

## Usage Instructions

In order to run the time measurement tools, specifically the `onnx_to_tensorrt.py` and `tensorrt_time_measure.py` tools, you need to build and run the tools under a specific docker environement provided by this [Dockerfile](../setting_environment/tensorrt/Dockerfile).

### 1. Measuring PyTorch Model Performance

```bash
python3 tools/performance_tools/torch_time_measure.py \
    <config_file> \
    <checkpoint_file> \
    --batch-size <batch_size> \
    --max-iter <iterations>
```

Example:
```bash
python3 tools/performance_tools/torch_time_measure.py \
    work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/yolox_s_tlr_416x416_pedcar_t4dataset.py \
    work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/best_mAP_epoch_300.pth \
    --batch-size 6 \
    --max-iter 100
```

### 2. Converting ONNX to TensorRT

```bash
python3 tools/performance_tools/onnx_to_tensorrt.py \
    <onnx_model_path> \
    <output_engine_path> \
    [--fp16] \
    [--workspace <workspace_size_in_GB>]
    [--max_dynamic_shape <max size for dynamic parameters>]
```

Example:
```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/performance_tools/onnx_to_tensorrt.py \
    work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/tlr_car_ped_yolox_s_batch_6.onnx \
    work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/tlr_car_ped_yolox_s_batch_6.engine \
    --workspace 4

CUDA_VISIBLE_DEVICES=1 python3 tools/performance_tools/onnx_to_tensorrt.py \
    work_dirs/centerpoint/pts_voxel_encoder.onnx work_dirs/centerpoint/pts_voxel_encoder.engine \
    --fp16
    --max_dynamic_shape 40000 32 9

CUDA_VISIBLE_DEVICES=1 python3 tools/performance_tools/onnx_to_tensorrt.py \
    work_dirs/centerpoint/pts_backbone_neck_head.onnx work_dirs/centerpoint/pts_backbone_neck_head.engine \
    --fp16
    --max_dynamic_shape 1 32 760 760
```

### 3. Measuring TensorRT Performance

```bash
python3 tools/performance_tools/tensorrt_time_measure.py \
    --engine_path <path_to_engine> \
    --iterations <num_iterations>
```

Example:
```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/performance_tools/tensorrt_time_measure.py \
    --engine_path work_dirs/yolox_s_tlr_416x416_pedcar_t4dataset/tlr_car_ped_yolox_s_batch_6.engine \
    --iterations 1000
```

## Notes

- Use `CUDA_VISIBLE_DEVICES` to specify which GPU to use
- The TensorRT conversion supports FP16 mode for faster inference, but it might cause slight drop in accuracy.
- Workspace size for TensorRT can be adjusted based on your model's requirements
- Make sure your ONNX model is compatible with TensorRT before conversion

## Common Issues

2. If TensorRT conversion fails:
   - Check ONNX model compatibility
   - Ensure all required operators are supported
   - Try updating TensorRT/ONNX versions
