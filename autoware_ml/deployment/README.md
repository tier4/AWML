# Autoware ML Deployment Framework

A unified, task-agnostic deployment framework for exporting, verifying, and evaluating machine learning models across different backends (ONNX, TensorRT).

## Architecture Overview

```
Deployment Framework
├── Core Abstractions
│   ├── BaseDataLoader       # Task-specific data loading
│   ├── BaseEvaluator        # Task-specific evaluation
│   ├── BaseBackend          # Unified inference interface
│   └── BaseDeploymentConfig # Configuration management
│
├── Backends
│   ├── PyTorchBackend       # PyTorch inference
│   ├── ONNXBackend          # ONNX Runtime inference
│   └── TensorRTBackend      # TensorRT inference
│
├── Exporters
│   ├── ONNXExporter         # PyTorch → ONNX
│   └── TensorRTExporter     # ONNX → TensorRT
│
└── Project Implementations
    ├── CalibrationStatusClassification/deploy/
    ├── YOLOX/deploy/  
    └── CenterPoint/deploy/  
```

---

## 🚀 Quick Start

### For Implemented Projects

#### CalibrationStatusClassification

```bash
python projects/CalibrationStatusClassification/deploy/main.py \
    projects/CalibrationStatusClassification/deploy/deploy_config.py \
    projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py \
    checkpoint.pth \
    --work-dir work_dirs/deployment
```

See `projects/CalibrationStatusClassification/deploy/README.md` for details.

---

## 📚 Documentation

- **Design Document**: `/docs/design/deploy_pipeline_design.md`
- **Architecture**: See above
- **Per-Project Guides**: `projects/{PROJECT}/deploy/README.md`

---

## 🔧 Development Guidelines

### Adding a New Project

1. **Create deploy directory**: `projects/{PROJECT}/deploy/`

2. **Implement DataLoader**:
   ```python
   from autoware_ml.deployment.core import BaseDataLoader

   class YourDataLoader(BaseDataLoader):
       def load_sample(self, index: int) -> Dict[str, Any]:
           # Load raw data
           pass

       def preprocess(self, sample: Dict[str, Any]) -> torch.Tensor:
           # Preprocess for model input
           pass

       def get_num_samples(self) -> int:
           return len(self.data)
   ```

3. **Implement Evaluator**:
   ```python
   from autoware_ml.deployment.core import BaseEvaluator

   class YourEvaluator(BaseEvaluator):
       def evaluate(self, model_path, data_loader, ...):
           # Run inference and compute metrics
           pass

       def print_results(self, results):
           # Pretty print results
           pass
   ```

4. **Create deployment config** (`deploy_config.py`)

5. **Create main script** (`main.py`)

6. **Test and document**
