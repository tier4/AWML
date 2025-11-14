# AWML Deployment Framework

A unified, task-agnostic deployment framework for exporting PyTorch models to ONNX and TensorRT, with comprehensive verification and evaluation capabilities.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project-Specific Implementations](#project-specific-implementations)
- [Pipeline Architecture](#pipeline-architecture)
- [Export Workflow](#export-workflow)
- [Verification & Evaluation](#verification--evaluation)

---

## Overview

The AWML Deployment Framework provides a standardized approach to model deployment across different projects (CenterPoint, YOLOX, CalibrationStatusClassification). It abstracts common deployment workflows while allowing project-specific customizations.

### Design Principles

1. **Unified Interface**: Single entry point (`DeploymentRunner`) for all deployment tasks
2. **Task-Agnostic Core**: Base classes that work across detection, classification, and segmentation
3. **Backend Flexibility**: Support for PyTorch, ONNX, and TensorRT backends
4. **Pipeline Architecture**: Shared preprocessing/postprocessing with backend-specific inference
5. **Configuration-Driven**: All settings controlled via config files

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Project Entry Points                       │
│  (projects/*/deploy/main.py)                           │
│  - CenterPoint, YOLOX-ELAN, Calibration                │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│           DeploymentRunner (Unified Runner)             │
│  - Coordinates export → verification → evaluation      │
│  - Manages model loading, export, verification         │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│   Exporters    │   │   Evaluators    │
│  - ONNX        │   │  - Task-specific│
│  - TensorRT    │   │  - Metrics      │
└────────────────┘   └─────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Pipeline Architecture                      │
│  - BaseDeploymentPipeline                               │
│  - Task-specific pipelines (Detection2D/3D, Classify)   │
│  - Backend-specific implementations                     │
└──────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **DeploymentRunner**
The unified runner that orchestrates the complete deployment workflow:

- **Model Loading**: Loads PyTorch models from checkpoints
- **Export**: Exports to ONNX and/or TensorRT
- **Verification**: Scenario-based verification across backends
- **Evaluation**: Performance metrics and latency statistics

#### 2. **Base Classes**

- **`BaseDeploymentConfig`**: Configuration container for deployment settings
- **`BaseEvaluator`**: Abstract interface for task-specific evaluation
- **`BaseDataLoader`**: Abstract interface for data loading
- **`BaseDeploymentPipeline`**: Abstract pipeline for inference

#### 3. **Exporters**

- **`ONNXExporter`**: Standard ONNX export with model wrapping support
- **`TensorRTExporter`**: TensorRT engine building with precision policies
- **Project-specific exporters**: Custom exporters for complex models (e.g., CenterPoint)

#### 4. **Pipelines**

- **`BaseDeploymentPipeline`**: Abstract base with `preprocess() → run_model() → postprocess()`
- **Task-specific pipelines**: `Detection2DPipeline`, `Detection3DPipeline`, `ClassificationPipeline`
- **Backend implementations**: PyTorch, ONNX, TensorRT variants for each pipeline

---

## Key Features

### 1. Unified Deployment Workflow

All projects follow the same workflow:
```
Load Model → Export ONNX → Export TensorRT → Verify → Evaluate
```

### 2. Scenario-Based Verification

Flexible verification system that compares outputs between backends:

```python
verification = dict(
    enabled=True,
    scenarios={
        "both": [
            {"ref_backend": "pytorch", "ref_device": "cpu",
             "test_backend": "onnx", "test_device": "cpu"},
            {"ref_backend": "onnx", "ref_device": "cpu",
             "test_backend": "tensorrt", "test_device": "cuda:0"},
        ]
    }
)
```

### 3. Multi-Backend Evaluation

Evaluate models across multiple backends with consistent metrics:

```python
evaluation = dict(
    enabled=True,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

### 4. Pipeline Architecture

Shared preprocessing/postprocessing with backend-specific inference:

- **Preprocessing**: Image resize, normalization, voxelization (shared)
- **Model Inference**: Backend-specific (PyTorch/ONNX/TensorRT)
- **Postprocessing**: NMS, coordinate transform, decoding (shared)

### 5. Flexible Export Modes

- `mode="onnx"`: Export PyTorch → ONNX only
- `mode="trt"`: Build TensorRT from existing ONNX
- `mode="both"`: Full export pipeline
- `mode="none"`: Skip export (evaluation only)

### 6. Precision Policies for TensorRT

Support for different TensorRT precision modes:

- `auto`: TensorRT decides automatically
- `fp16`: FP16 precision
- `fp32_tf32`: FP32 with TF32 acceleration
- `explicit_int8`: INT8 quantization

---

## Usage

### Basic Usage

```bash
# CenterPoint deployment
python projects/CenterPoint/deploy/main.py \
    configs/deploy_config.py \
    configs/model_config.py \
    checkpoint.pth

# YOLOX deployment
python projects/YOLOX_opt_elan/deploy/main.py \
    configs/deploy_config.py \
    configs/model_config.py \
    checkpoint.pth

# Calibration deployment
python projects/CalibrationStatusClassification/deploy/main.py \
    configs/deploy_config.py \
    configs/model_config.py \
    checkpoint.pth
```

### Command-Line Arguments

```bash
python deploy/main.py \
    <deploy_cfg> \          # Deployment configuration file
    <model_cfg> \           # Model configuration file
    [checkpoint] \          # Optional: checkpoint path (can be in config)
    --work-dir <dir> \      # Optional: override work directory
    --device <device> \     # Optional: override device
    --log-level <level>     # Optional: logging level (DEBUG, INFO, WARNING, ERROR)
```

### Export Modes

#### Export ONNX Only
```python
export = dict(
    mode="onnx",
    checkpoint_path="model.pth",
    work_dir="work_dirs/deployment",
)
```

#### Build TensorRT from Existing ONNX
```python
export = dict(
    mode="trt",
    onnx_path="work_dirs/deployment/onnx/model.onnx",
    work_dir="work_dirs/deployment",
)
```

#### Full Export Pipeline
```python
export = dict(
    mode="both",
    checkpoint_path="model.pth",
    work_dir="work_dirs/deployment",
)
```

#### Evaluation Only (No Export)
```python
export = dict(
    mode="none",
    work_dir="work_dirs/deployment",
)
```

---

## Configuration

### Configuration Structure

```python
# Task type
task_type = "detection3d"  # or "detection2d", "classification"

# Export configuration
export = dict(
    mode="both",  # "onnx", "trt", "both", "none"
    work_dir="work_dirs/deployment",
    checkpoint_path="model.pth",
    onnx_path=None,  # Optional: for mode="trt"
)

# Runtime I/O settings
runtime_io = dict(
    info_file="data/info.pkl",  # Dataset info file
    sample_idx=0,  # Sample index for export
)

# Model I/O configuration
model_io = dict(
    input_name="input",
    input_shape=(3, 960, 960),  # (C, H, W)
    input_dtype="float32",
    output_name="output",
    batch_size=1,  # or None for dynamic
    dynamic_axes={...},  # When batch_size=None
)

# ONNX configuration
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    save_file="model.onnx",
    multi_file=False,  # True for multi-file ONNX (e.g., CenterPoint)
)

# Backend configuration
backend_config = dict(
    common_config=dict(
        precision_policy="auto",  # "auto", "fp16", "fp32_tf32", "explicit_int8"
        max_workspace_size=1 << 30,  # 1 GB
    ),
)

# Verification configuration
verification = dict(
    enabled=True,
    num_verify_samples=3,
    tolerance=0.1,
    devices={
        "cpu": "cpu",
        "cuda": "cuda:0",
    },
    scenarios={
        "both": [
            {"ref_backend": "pytorch", "ref_device": "cpu",
             "test_backend": "onnx", "test_device": "cpu"},
        ]
    }
)

# Evaluation configuration
evaluation = dict(
    enabled=True,
    num_samples=100,  # or -1 for all samples
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

### Configuration Examples

See project-specific configs:
- `projects/CenterPoint/deploy/configs/deploy_config.py`
- `projects/YOLOX_opt_elan/deploy/configs/deploy_config.py`
- `projects/CalibrationStatusClassification/deploy/configs/deploy_config.py`

---

## Project-Specific Implementations

### CenterPoint (3D Detection)

**Features:**
- Multi-file ONNX export (voxel encoder + backbone/head)
- ONNX-compatible model configuration
- Custom exporters for complex model structure

**Key Files:**
- `projects/CenterPoint/deploy/main.py`
- `projects/CenterPoint/deploy/evaluator.py`
- `autoware_ml/deployment/pipelines/centerpoint/`

**Pipeline Structure:**
```
preprocess() → run_voxel_encoder() → process_middle_encoder() →
run_backbone_head() → postprocess()
```

### YOLOX (2D Detection)

**Features:**
- Standard single-file ONNX export
- Model wrapper for ONNX-compatible output format
- ReLU6 → ReLU replacement for ONNX compatibility

**Key Files:**
- `projects/YOLOX_opt_elan/deploy/main.py`
- `projects/YOLOX_opt_elan/deploy/evaluator.py`
- `autoware_ml/deployment/pipelines/yolox/`

**Pipeline Structure:**
```
preprocess() → run_model() → postprocess()
```

### CalibrationStatusClassification

**Features:**
- Binary classification task
- Simple single-file ONNX export
- Calibrated/miscalibrated data loader variants

**Key Files:**
- `projects/CalibrationStatusClassification/deploy/main.py`
- `projects/CalibrationStatusClassification/deploy/evaluator.py`
- `autoware_ml/deployment/pipelines/calibration/`

**Pipeline Structure:**
```
preprocess() → run_model() → postprocess()
```

---

## Pipeline Architecture

### Base Pipeline

All pipelines inherit from `BaseDeploymentPipeline`:

```python
class BaseDeploymentPipeline(ABC):
    @abstractmethod
    def preprocess(self, input_data, **kwargs) -> Any:
        """Preprocess input data"""
        pass

    @abstractmethod
    def run_model(self, preprocessed_input) -> Any:
        """Backend-specific model inference"""
        pass

    @abstractmethod
    def postprocess(self, model_output, metadata) -> Any:
        """Postprocess model output"""
        pass

    def infer(self, input_data, **kwargs):
        """Complete inference pipeline"""
        preprocessed = self.preprocess(input_data, **kwargs)
        model_output = self.run_model(preprocessed)
        predictions = self.postprocess(model_output, metadata)
        return predictions
```

### Task-Specific Pipelines

#### Detection2DPipeline
- Shared preprocessing: image resize, normalization, padding
- Shared postprocessing: bbox decoding, NMS, coordinate transform
- Backend-specific: model inference

#### Detection3DPipeline
- Shared preprocessing: voxelization, feature extraction
- Shared postprocessing: 3D bbox decoding, NMS
- Backend-specific: voxel encoder, backbone/head inference

#### ClassificationPipeline
- Shared preprocessing: image normalization
- Shared postprocessing: softmax, top-k selection
- Backend-specific: model inference

### Backend Implementations

Each pipeline has three backend implementations:

1. **PyTorch Pipeline**: Direct PyTorch model inference
2. **ONNX Pipeline**: ONNX Runtime inference
3. **TensorRT Pipeline**: TensorRT engine inference

Example for YOLOX:
- `YOLOXPyTorchPipeline`: Uses PyTorch model directly
- `YOLOXONNXPipeline`: Uses ONNX Runtime
- `YOLOXTensorRTPipeline`: Uses TensorRT engine

---

## Export Workflow

### ONNX Export

1. **Model Preparation**: Load PyTorch model, apply model wrapper if needed
2. **Input Preparation**: Get sample input from data loader
3. **Export**: Call `torch.onnx.export()` with configured settings
4. **Simplification**: Optional ONNX model simplification
5. **Save**: Save to `work_dir/onnx/`

### TensorRT Export

1. **ONNX Validation**: Verify ONNX model exists
2. **Network Creation**: Parse ONNX, create TensorRT network
3. **Precision Configuration**: Apply precision policy flags
4. **Optimization Profile**: Configure input shape ranges
5. **Engine Building**: Build and save TensorRT engine
6. **Save**: Save to `work_dir/tensorrt/`

### Multi-File Export (CenterPoint)

CenterPoint uses a multi-file ONNX structure:
- `voxel_encoder.onnx`: Voxel feature extraction
- `backbone_head.onnx`: Backbone and detection head

The exporter handles:
- Sequential export of each component
- Proper input/output linking
- Directory-based organization

---

## Verification & Evaluation

### Verification

Policy-based verification compares outputs between backends:

```python
# Verification scenarios example
verification = dict(
    enabled=True,
    scenarios={
        "both": [
            {
                "ref_backend": "pytorch",
                "ref_device": "cpu",
                "test_backend": "onnx",
                "test_device": "cpu"
            }
        ]
    },
    tolerance=0.1,  # Maximum allowed difference
    num_verify_samples=3
)
```

**Verification Process:**
1. Load reference model (PyTorch or ONNX)
2. Load test model (ONNX or TensorRT)
3. Run inference on same samples
4. Compare outputs with tolerance
5. Report pass/fail for each sample

### Evaluation

Task-specific evaluation with consistent metrics:

**Detection Tasks:**
- mAP (mean Average Precision)
- Per-class AP
- Latency statistics

**Classification Tasks:**
- Accuracy
- Precision, Recall
- Per-class metrics
- Confusion matrix
- Latency statistics

**Evaluation Configuration:**
```python
evaluation = dict(
    enabled=True,
    num_samples=100,  # or -1 for all
    verbose=False,
    backends={
        "pytorch": {"enabled": True, "device": "cpu"},
        "onnx": {"enabled": True, "device": "cpu"},
        "tensorrt": {"enabled": True, "device": "cuda:0"},
    }
)
```

---

## File Structure

```
autoware_ml/deployment/
├── core/                          # Core base classes
│   ├── base_config.py            # Configuration management
│   ├── base_data_loader.py        # Data loader interface
│   ├── base_evaluator.py          # Evaluator interface
│   └── preprocessing_builder.py   # Preprocessing builder
│
├── exporters/                     # Model exporters
│   ├── base_exporter.py           # Exporter base class
│   ├── onnx_exporter.py           # ONNX exporter
│   ├── tensorrt_exporter.py       # TensorRT exporter
│   └── model_wrappers.py          # Model wrappers for ONNX
│
├── pipelines/                     # Task-specific pipelines
│   ├── base/                      # Base pipeline classes
│   │   ├── base_pipeline.py       # Pipeline base class
│   │   ├── detection_2d_pipeline.py   # 2D detection pipeline
│   │   ├── detection_3d_pipeline.py   # 3D detection pipeline
│   │   └── classification_pipeline.py # Classification pipeline
│   ├── centerpoint/               # CenterPoint pipelines
│   │   ├── centerpoint_pipeline.py
│   │   ├── centerpoint_pytorch.py
│   │   ├── centerpoint_onnx.py
│   │   └── centerpoint_tensorrt.py
│   ├── yolox/                     # YOLOX pipelines
│   │   ├── yolox_pipeline.py
│   │   ├── yolox_pytorch.py
│   │   ├── yolox_onnx.py
│   │   └── yolox_tensorrt.py
│   └── calibration/               # Calibration pipelines
│       ├── calibration_pipeline.py
│       ├── calibration_pytorch.py
│       ├── calibration_onnx.py
│       └── calibration_tensorrt.py
│
└── runners/                       # Deployment runners
    └── deployment_runner.py       # Unified deployment runner
projects/
├── CenterPoint/deploy/
│   ├── main.py                    # Entry point
│   ├── evaluator.py               # CenterPoint evaluator
│   ├── data_loader.py             # CenterPoint data loader
│   └── configs/
│       └── deploy_config.py       # Deployment configuration
│
├── YOLOX_opt_elan/deploy/
│   ├── main.py
│   ├── evaluator.py
│   ├── data_loader.py
│   └── configs/
│       └── deploy_config.py
│
└── CalibrationStatusClassification/deploy/
    ├── main.py
    ├── evaluator.py
    ├── data_loader.py
    └── configs/
        └── deploy_config.py
```

---

## Best Practices

### 1. Configuration Management

- Keep deployment configs separate from model configs
- Use relative paths for data files
- Document all configuration options

### 2. Model Export

- Always verify ONNX export before TensorRT conversion
- Use appropriate precision policies for TensorRT
- Test with multiple samples during export

### 3. Verification

- Start with small tolerance (0.01) and increase if needed
- Verify on representative samples
- Check both accuracy and numerical differences

### 4. Evaluation

- Use consistent evaluation settings across backends
- Report latency statistics (mean, std, min, max)
- Compare metrics across backends

### 5. Pipeline Development

- Inherit from appropriate base pipeline
- Share preprocessing/postprocessing logic
- Keep backend-specific code minimal

---

## Troubleshooting

### Common Issues

1. **ONNX Export Fails**
   - Check model compatibility (unsupported ops)
   - Verify input shapes match model expectations
   - Try different opset versions

2. **TensorRT Build Fails**
   - Verify ONNX model is valid
   - Check input shape configuration
   - Reduce workspace size if memory issues

3. **Verification Fails**
   - Check tolerance settings
   - Verify same preprocessing for all backends
   - Check device compatibility

4. **Evaluation Errors**
   - Verify data loader paths
   - Check model output format
   - Ensure correct task type in config

---

## Future Enhancements

- [ ] Support for more task types (segmentation, etc.)
- [ ] Automatic precision tuning for TensorRT
- [ ] Distributed evaluation support
- [ ] Integration with MLOps pipelines
- [ ] Performance profiling tools

---

## Contributing

When adding a new project:

1. Create project-specific evaluator and data loader
2. Implement task-specific pipeline (if needed)
3. Create deployment configuration
4. Add entry point script
5. Update documentation

---

## License

See LICENSE file in project root.
