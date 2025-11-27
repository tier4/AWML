# tools/segmentation3d

The pipeline to develop models for 3D LiDAR semantic segmentation.

**Note on dataset preparation**: Dataset info file (`.pkl`) generation is handled by the script in [tools/detection3d](/tools/detection3d/README.md), which processes all annotation types including semantic segmentation labels when available.

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S
- Supported dataset
  - [x] NuScenes with 3D semantic segmentation
  - [x] T4dataset with 3D semantic segmentation
- Other supported feature
  - [ ] Add unit test

## 1. Setup environment

See [tutorial_installation](/docs/tutorial/tutorial_installation.md) to set up the environment.

## 2. Prepare T4dataset

- Run docker

```sh
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

- Make info files for T4dataset j6gen2 lidarseg
  - This process takes time.

```sh
python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/segmentation3d/dataset/t4dataset/j6gen2_base.py --version j6gen2_lidarseg --max_sweeps 1 --out_dir ./data/t4dataset/info/user_name
```

## 3. Train

TODO

## 4. Analyze

TODO

### 4.2. Visualization

TODO

## 5. Deploy

TODO
