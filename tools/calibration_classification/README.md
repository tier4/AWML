# tools/calibration_classification

The pipeline to make the model.
It contains training, evaluation, and visualization for Calibration classification.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- Supported dataset
  - [] NuScenes
  - [x] T4dataset
- Other supported feature
  - [ ] Add unit test

## 1. Setup environment

Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md) to set up the environment.

## 2. Prepare dataset

Prepare the dataset you use.

### 2.1. T4dataset

- Run docker

```sh
docker run -it --rm --gpus --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for T4dataset X2 Gen2

```sh
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/
```

**Available command line arguments:**

- `--config` (required): Path to the T4dataset configuration file
- `--root_path` (required): Root path of the dataset
- `--version` (required): Product version (e.g., x2)
- `-o, --out_dir` (required): Output directory for info files
- `--lidar_channel` (optional): Lidar channel name (default: LIDAR_CONCAT)
- `--target_cameras` (optional): List of target cameras to process (default: all cameras)

**Examples:**

```sh
# Basic usage with default settings (all cameras)
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/

# Process only specific cameras
python tools/calibration_classification/create_data_t4dataset.py --config /workspace/autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py --version gen2_base --root_path ./data/t4dataset -o ./data/t4dataset/calibration_info/ --target_cameras CAM_FRONT CAM_LEFT CAM_RIGHT
```

**Output files:**
The script generates three pickle files for train/val/test splits:
- `t4dataset_{version}_infos_train.pkl`
- `t4dataset_{version}_infos_val.pkl`
- `t4dataset_{version}_infos_test.pkl`

Each file contains calibration information including:
- Camera and LiDAR data paths
- Transformation matrices (lidar2cam, cam2ego, etc.)
- Timestamps and metadata

**Example data structure:**
```python
{
    'frame_id': 'CAM_FRONT',
    'frame_idx': '00003',
    'image': {
        'cam2ego': [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]],
        'cam2img': [[1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0]],
        'cam_pose': [[1.0, 0.0, 0.0, 100.0],
                     [0.0, 1.0, 0.0, 200.0],
                     [0.0, 0.0, 1.0, 1.5],
                     [0.0, 0.0, 0.0, 1.0]],
        'height': 1080,
        'img_path': 'data/camera/front/00003.jpg',
        'lidar2cam': [[1.0, 0.0, 0.0, 0.1],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]],
        'distortion_coefficients': [0.0, 0.0, 0.0, 0.0, 0.0],
        'sample_data_token': 'sample_token_123',
        'timestamp': 1234567890,
        'width': 1920
    },
    'lidar_points': {
        'lidar2ego': [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]],
        'lidar_path': 'data/lidar/00003.pcd.bin',
        'lidar_pose': [[1.0, 0.0, 0.0, 100.0],
                       [0.0, 1.0, 0.0, 200.0],
                       [0.0, 0.0, 1.0, 1.5],
                       [0.0, 0.0, 0.0, 1.0]],
        'sample_data_token': 'lidar_token_456',
        'timestamp': 1234567890
    },
    'sample_idx': 3,
    'scene_id': 'scene_001'
}
```

**Key fields explained:**
- `frame_id`: Camera identifier (e.g., CAM_FRONT, CAM_LEFT)
- `frame_idx`: Frame index in the sequence
- `image`: Camera-specific data including intrinsic/extrinsic parameters
- `lidar_points`: LiDAR data with transformation matrices
- `sample_idx`: Sample index in the dataset
- `scene_id`: Scene identifier

### 2.2. Calibration Visualization (Before training)

- Visualization for calibration and image view
  - This tool visualizes the calibration between LiDAR and camera sensors
  - It shows the projection of LiDAR points onto camera images using info.pkl files

```sh
# Process all samples from info.pkl
python tools/calibration_classification/visualize_lidar_camera_projection.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/info.pkl --data_root data/ --output_dir ./work_dirs/calibration_visualization

# Process specific sample
python tools/calibration_classification/visualize_lidar_camera_projection.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/info.pkl --data_root data/ --output_dir ./work_dirs/calibration_visualization --sample_idx 0

# Process specific indices
python tools/calibration_classification/visualize_lidar_camera_projection.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/info.pkl --data_root data/ --output_dir ./work_dirs/calibration_visualization --indices 0 1 2
```

- For T4dataset visualization:

```sh
python tools/calibration_classification/visualize_lidar_camera_projection.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py --info_pkl data/t4dataset/calibration_info/t4dataset_gen2_base_infos_test.pkl --data_root data/t4dataset --output_dir ./work_dirs/calibration_visualization
```

## 3. Visualization Settings (During training, validation, testing)

Understanding visualization configuration is crucial for calibration classification tasks. The `CalibrationClassificationTransform` controls two main visualization types:

### 3.1. Visualization Types

**Projection Visualization** (`projection_vis_dir`):
- Shows LiDAR points projected onto camera images
- Helps verify calibration accuracy between sensors

**Results Visualization** (`results_vis_dir`):
- Displays model predictions
- Shows classification results overlaid on images

### 3.2. Configuration Example

```python
# In config file
test_projection_vis_dir = "./test_projection_vis_t4dataset/"
test_results_vis_dir = "./test_results_vis_t4dataset/"

# In transform pipeline
dict(
    type="CalibrationClassificationTransform",
    mode="test",
    undistort=True,
    enable_augmentation=False,
    data_root=data_root,
    projection_vis_dir=test_projection_vis_dir,  # LiDAR projection visualization
    results_vis_dir=test_results_vis_dir,        # Model prediction visualization
),
```

### 3.3. Usage Strategy

**Training/Validataion Phase:**
- Disable visualization for efficiency: `projection_vis_dir=None`, `results_vis_dir=None`
- Focus on model training performance

**Testing Phase:**
- Enable visualization for analysis: Set both directories to desired paths
- Use projection visualization to verify calibration quality
- Use results visualization to understand model predictions

**Output Files:**
- `projection_vis_dir/`: LiDAR points overlaid on camera images
- `results_vis_dir/`: Classification results with predicted labels

## 4. Train
### 4.1. Environment set up
Set `CUBLAS_WORKSPACE_CONFIG` for the deterministic behavior, plese check this [nvidia doc](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) for more info

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 4.2. Train

- Train in general by below command.

```sh
python tools/calibration_classification/train.py {config_file}
```

* Example
```sh
python tools/calibration_classification/train.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py
```

- You can use docker command for training as below.

```sh
docker run -it --rm --gpus --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/calibration_classification/train.py {config_file}'
```

### 4.3. Log analysis by Tensorboard

- Run the TensorBoard and navigate to http://127.0.0.1:6006/

```sh
tensorboard --logdir work_dirs --bind_all
```

## 5. Analyze
### 5.1. Evaluation

- Evaluation

```sh
python tools/calibration_classification/test.py {config_file} {checkpoint_file}
```
* Example
```sh
python tools/calibration_classification/test.py projects/CalibrationStatusClassification/configs/t4dataset/resnet18_5ch_1xb16-50e_j6gen2.py  epoch_25.pth --out {output_file}
```
