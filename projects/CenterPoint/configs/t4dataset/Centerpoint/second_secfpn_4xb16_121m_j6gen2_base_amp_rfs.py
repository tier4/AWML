_base_ = [
    "second_secfpn_4xb16_121m_j6gen2_base_amp.py",
]

experiment_group_name = "second_secfpn_4xb16_121m_j6gen2_base_amp_rfs"

# user setting
work_dir = _base_.experiment_base_work_dir + "/" + experiment_group_name + "/"

train_frame_object_sampler = dict(
    type="FrameObjectSampler",
    object_samplers=[
        dict(
            type="ObjectBEVDistanceSampler",
            bev_distance_thresholds=[
                _base_.point_cloud_range[0],
                _base_.point_cloud_range[1],
                _base_.point_cloud_range[3],
                _base_.point_cloud_range[4],
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
        repeat_sampling_factor=0.30,
        frame_object_sampler=train_frame_object_sampler,
    ),
)
