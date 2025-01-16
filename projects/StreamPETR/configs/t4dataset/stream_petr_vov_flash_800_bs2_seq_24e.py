_base_ = [
    "../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py",
]
custom_imports = dict(
    imports=["projects.StreamPETR.stream_petr"],
    allow_failed_imports=False,
)
custom_imports["imports"] += _base_.custom_imports["imports"]

backbone_norm_cfg = dict(type='LN', requires_grad=True)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False) # fix img_norm
# For nuScenes we usually do 10-class detection
class_names = _base_.class_names

num_gpus = 1
batch_size = 2
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24
backend_args = None

eval_class_range = {
    "car": 120,
    "truck": 120,
    "bus": 120,
    "bicycle": 120,
    "pedestrian": 120,
}

queue_length = 4
num_frame_losses = 2
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP', ###use checkpoint to save memory
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    img_roi_head=dict(
        type='mmdet.FocalHead',
        num_classes=10,
        in_channels=256,
        bbox_coder = dict(type='mmdet.DistancePointBBoxCoder'),
        loss_cls = dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls2d=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='mmdet.HungarianAssigner2D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.),
                reg_cost=dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='mmdet.BBox3DL1Cost', weight=10.0)))
        ),
    pts_bbox_head=dict(
        type='StreamPETRHead',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='mmdet.NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        train_cfg=dict(pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='mmdet.HungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='mmdet.BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='mmdet.IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range),
            ))
        ),
    
    )

data_root = "./data/"
info_directory_path = "info/cameraonly/streampetr/"

file_client_args = dict(backend='disk')


ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='StreamPETRLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='mmdet.ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='mmdet.GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(type='mmdet.NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='mmdet.PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='StreamPETRLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='mmdet.ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='mmdet.NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='mmdet.PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="StreamPETRDataset",
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=train_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        test_mode=False,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        random_length=1,
        queue_length=queue_length,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="StreamPETRDataset",
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        data_prefix=_base_.data_prefix,
        test_mode=False,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="StreamPETRDataset",
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        data_prefix=_base_.data_prefix,
        test_mode=False,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)


val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
)
test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
)


train_cfg = dict(by_epoch=True, max_epochs=num_epochs)
val_cfg = dict()
test_cfg = dict()

optimizer = dict(
    type='AdamW', 
    lr=4e-4, # bs 8: 2e-4 || bs 16: 4e-4,
    weight_decay=0.01)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, dtype='float16')
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
# load_from='work_dirs/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
load_from=None
resume_from=None
