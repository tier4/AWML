# Model name
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: [package_name](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/)
- Supported dataset
  - [ ] COCO dataset
  - [ ] T4dataset
- Other supported feature
  - [ ] Add script to make .onnx file and deploy to Autoware
  - [ ] Add unit test
- Limited feature

## Results and models

- See [mmdet](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox) and download pretrain model.

## Get started
### 1. Setup

TBD

### 2. config

TBD

### 3. Train

TBD

### 4. Deploy

TBD

### 5. Inference

- For an image

```sh
python tools/detection2d/image_demo.py {image_path} \
projects/YOLOX/configs/yolox_l_8xb8-300e_coco.py \
--weights work_dirs/pretrain/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
--out-dir work_dirs/yolox
```

- For directory

```sh
python tools/detection2d/image_demo.py data/test_data/ \
projects/YOLOX/configs/yolox_l_8xb8-300e_coco.py \
--weights work_dirs/pretrain/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
--out-dir work_dirs/yolox
```

## Troubleshooting

## Reference
