_base_ = [
    "second_secfpn_4xb16_121m_j6gen2_base_amp.py",
]

# user setting
work_dir = "work_dirs/centerpoint_2_6/" + _base_.dataset_type + "/second_secfpn_4xb16_121m_j6gen2_base_amp_rfs_from_pretrain/"

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
            target_label_index=4  # 4 is the label index for pedestrian 
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

load_from = "work_dirs/centerpoint_2_6/T4Dataset/second_secfpn_4xb16_121m_base_amp_rfs/epoch_48.pth"