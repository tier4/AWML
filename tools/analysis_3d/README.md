# analysis_3d

It provides a framework to developers in `AWML` to add analyses for 3D annotations in T4dataset easily.
With this framework, developers don't need to generate any `info` files or rewrite their data loading for the dataset.
They only need to follow `AnalysisCallbackInterface` to add the analyses they are interested in.

## Summary

- [Support priority](https://github.com/tier4/AWML/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier B
- Supported dataset
  - [x] T4dataset
  - [] NuScenes
- Other supported feature
  - [x] Distribution of categories
  - [x] Distribution of attributes in each category
  - [ ] Distribution of sizes/orientation
  - [ ] Add unit tests

## Get started
### 1. Setup

- Please follow the [installation tutorial](/docs/tutorial/tutorial_detection_3d.md) to set up the environment.
- Run docker

```sh
docker run -it --rm --gpus all --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml
```

### 2. Analysis
#### 2.1. Dataset analysis

Make sure the dataset follows the [T4dataset format](https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md), note that it doesn't need any `info` file

```sh
# T4dataset (base)
python tools/analysis_3d/run.py --config_path autoware_ml/configs/detection3d/dataset/t4dataset/base.py --data_root_path data/t4dataset/ --out_dir data/t4dataset/analyses/
```

## For developer

1. Add a new analysis to inherit `AnalysisCallbackInterface` as a callback, and implement `run()`, for example, `tools/analysis_3d/callbacks/category_attribute.py`
2. Import the new analysis in `AnalysisRunner`, and add them to the list of `analysis_callbacks`, for example,

```python
self.analysis_callbacks: List[AnalysisCallbackInterface] = [
    ...
    CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name='bicycle',
                analysis_dir='remapping_bicycle_attr',
                remapping_classes=self.remapping_classes),
    # This is the new CategoryAttributeAnalysisCallback
    CategoryAttributeAnalysisCallback(
      out_path=self.out_path,
      category_name='vehicle.bus',
      analysis_dir='vehicle_bus_attr'
    ),
]
```

## References
