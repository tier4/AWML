# Samplers
This folder conists of `Sampler` implementation

#### 2.3 Train CenterPoint model with Repeat Factor Sampling (RFS)
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
