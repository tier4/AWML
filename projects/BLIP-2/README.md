# BLIP-2
## Summary

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- ROS package: Not supported
- Supported dataset for training: Not supported
- Other supported feature
  - [ ] Add unit test
- Limited feature

## Results and models

- See [BLIP-2 of mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/main/configs/blip2).

## Get started
### 1. Setup

- 1.1. [Run setup environment at first](/tools/setting_environment/).
- 1.2. Run docker

```
docker run -it --rm --gpus all --shm-size=64g --name awml -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

### 2. Inference

```sh
# QA task
python projects/BLIP-2/demo.py "blip2-opt2.7b_3rdparty-zeroshot_vqa" {image_file} "What is this?"

# Image caption
python projects/BLIP-2/demo.py "blip2-opt2.7b_3rdparty-zeroshot_caption" {image_file}
```

## Troubleshooting

## Reference

- [BLIP-2 of mmpretrain](https://github.com/open-mmlab/mmpretrain/tree/main/configs/blip2).
