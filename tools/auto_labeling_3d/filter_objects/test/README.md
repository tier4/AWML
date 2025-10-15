# Filter Objects Test Suite

Comprehensive test suite for ThresholdFilter and NMSEnsembleModel classes.

## Overview

This test suite validates two key components:
- **ThresholdFilter**: Confidence-based filtering of 3D object detection results
- **NMSEnsembleModel**: Non-Maximum Suppression ensemble for multiple model predictions

## Test Structure

```
test/
├── conftest.py              # Shared fixtures and test data
├── test_threshold_filter.py # ThresholdFilter tests
├── test_nms_ensemble_model.py # NMSEnsembleModel tests
└── README.md               # This file
```

## ThresholdFilter Tests

ThresholdFilter removes low-confidence predictions using configurable thresholds per class.

### Algorithm Flow

```mermaid
flowchart TD
    A[Input Predictions] --> B{Check Confidence}
    B -->|Score ≥ Threshold| C[Keep Instance]
    B -->|Score < Threshold| D[Filter Out]
    C --> E[Output Results]
    D --> E
```

### Test Categories

```
Input: [car: 0.8, car: 0.3, pedestrian: 0.7, bicycle: 0.2]
Thresholds: {car: 0.5, pedestrian: 0.6, bicycle: 0.4}
                           ↓
Output: [car: 0.8, pedestrian: 0.7] # car:0.3 and bicycle:0.2 filtered out
```

- **Basic Filtering**: Confidence threshold application (≥ threshold kept)
- **Boundary Values**: Edge cases at exact threshold values
- **Empty Data**: Handling of empty input/output scenarios
- **Multi-frame**: Processing multiple time frames
- **Unknown Classes**: Graceful handling of unmapped classes
- **Statistics Logging**: Verification of filtering statistics

## NMSEnsembleModel Tests

NMSEnsembleModel combines predictions from multiple models using Non-Maximum Suppression.

### Algorithm Flow

```mermaid
flowchart TD
    A[Model 1 Predictions] --> D[Apply Weights for Selection]
    B[Model 2 Predictions] --> E[Apply Weights for Selection]
    C[Model N Predictions] --> F[Apply Weights for Selection]
    
    D --> G[Combine All Predictions]
    E --> G
    F --> G
    
    G --> H[Sort by Weighted Score]
    H --> I[Calculate IoU]
    I --> J{IoU > Threshold?}
    J -->|Yes| K[Select Higher Weighted Score]
    J -->|No| L[Keep Both]
    K --> M[Return Original Instances]
    L --> M
```

**Note**: Weighted scores are used for NMS selection only. Original confidence scores are preserved in the output instances.

### Ensemble Process Visualization

#### Case 1: High IoU (Overlap) - Suppression

![Ensemble Overlap Visualization](./nms_ensemble_with_overlap.svg)

This visualization shows how two overlapping bounding boxes from different models are processed through the NMS ensemble algorithm. The weighted scores (0.4 vs 0.3) are used only for NMS selection decisions, while the original confidence scores (0.8) are preserved in the final output instances.

#### Case 2: Low IoU (No Overlap) - Keep Both
![Ensemble No Overlap Visualization](./nms_ensemble_no_overlap.svg)

This visualization demonstrates the case where two detections from different models have low IoU (0.2 < 0.5), resulting in both detections being kept in the final output with their original confidence scores preserved.

### Test Categories

#### 1. NMSModelInstances
- Instance filtering and weighting per model
- Empty result handling

#### 2. Ensemble Logic

```mermaid
graph TD
    subgraph "3D Bounding Box Overlap"
        A["Box 1: [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]<br/>Score: 0.8"]
        B["Box 2: [1.1, 2.1, 0.0, 4.0, 2.0, 1.5, 0.1]<br/>Score: 0.6"]
        C["IoU = 0.863"]
    end
    
    subgraph "NMS Decisions"
        D["Threshold 0.5<br/>0.863 > 0.5<br/>→ Keep 1 box"]
        E["Threshold 0.9<br/>0.863 < 0.9<br/>→ Keep 2 boxes"]
    end
    
    A --> C
    B --> C
    C --> D
    C --> E
```

### Bird's Eye View IoU Calculation

```mermaid
graph LR
    subgraph "BEV Projection"
        A["3D Box → 2D Rectangle<br/>[x, y, dx, dy]"]
    end
    
    subgraph "IoU Formula"
        B["Intersection Area<br/>÷<br/>Union Area"]
    end
    
    subgraph "Test Cases"
        C["Identical: IoU = 1.0<br/>Partial: IoU = 0.863<br/>No overlap: IoU = 0.0"]
    end
    
    A --> B
    B --> C
```

- Single/multiple model ensemble
- Overlapping prediction handling
- Label group processing
- Empty prediction scenarios
- Weight validation

#### 3. Helper Functions
- **IoU Calculation**: Bird's Eye View intersection over union
- **NMS Algorithm**: Non-maximum suppression with various thresholds

### Key Test Data

```python
# Sample 3D bounding box [x, y, z, dx, dy, dz, yaw]
Box A: [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]
Box B: [1.1, 2.1, 0.0, 4.0, 2.0, 1.5, 0.1]  # IoU ≈ 0.863
```

## Running Tests

```bash
# All tests
python -m pytest tools/auto_labeling_3d/filter_objects/test/ -v

# Specific component
python -m pytest tools/auto_labeling_3d/filter_objects/test/test_threshold_filter.py -v
python -m pytest tools/auto_labeling_3d/filter_objects/test/test_nms_ensemble_model.py -v
```
