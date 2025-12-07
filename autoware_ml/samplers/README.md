# Samplers
This folder conists of custom implementations for `Sampler`. `Sampler` is a class to implement different sampling strategies in an experiment.  

### 2.1 DistributedWeightedRandomSampler
`DistributedWeightedRandomSampler` is the implementation of `WeightedRandomSampler` in a distributed manner across several gpus. `Dataloader` will select frame randomly based on their `weights`, and the weights can be determined by either initialization or `frame_weights` from a `Dataset`. By deafult, frames with higher weights are possible to be selected more than once in an epoch.  

### 2.2 ObjectSampler
`ObjectSampler` is the base class to sample certain objects in frame-level, for example, sample pedestrians that are close to the ego vehicle. To implement a new `ObjectSampler`, please inherit `ObjectSampler` and define the strategy in the function: `sample`

#### 2.3 Repeat Factor Sampling (RFS)
- To use RFS in an experiment, please use dataset: `T4FrameSamplerDataset`, `DistributedWeightedRandomSampler`, and add corresponding `ObjectSampler` to `T4FrameSamplerDataset`, for example:

```python
train_frame_object_sampler = dict(
    type="FrameObjectSampler",
    object_samplers=[
        dict(
            type="ObjectBEVDistanceSampler",
            bev_distance_thresholds=[
                point_cloud_range[0],
                point_cloud_range[1],
                point_cloud_range[3],
                point_cloud_range[4],
            ],
        ),
        dict(
            type="LowPedestriansObjectSampler",
            height_threshold=1.5,
            bev_distance_thresholds=[
                -50.0,
                -50.0,
                50.0,
                50.0,
            ],
        ),
    ],
)

train_dataloader = dict(
    sampler=dict(type="DistributedWeightedRandomSampler", shuffle=True),
    dataset=dict(
        type="T4FrameSamplerDataset",
        pipeline=train_pipeline,
        modality=_base_.input_modality,
        backend_args=_base_.backend_args,
        data_root=_base_.data_root,
        ann_file=_base_.info_directory_path + _base_.info_train_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        repeat_sampling_factor=0.30,
        frame_object_sampler=train_frame_object_sampler,
    ),
)
```
