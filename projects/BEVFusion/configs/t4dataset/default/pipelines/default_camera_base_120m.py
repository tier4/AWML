## This config is for the camera_base only model, without lidar points

_base_ = [
    "./default_lidar_120m.py",
]
input_modality = dict(use_lidar=True, use_camera=True)

# Image parameters
image_size = [384, 768]  # Height, Width
camera_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"]

train_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        to_float32=True,
        color_type="color",
        backend_args=_base_.backend_args,
        camera_order=camera_order,
    ),
    # We keep loading LiDAR points to make downstream BEV augmentation easier 
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_load_dim,
        backend_args=_base_.backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        # resize_lim=[0.28, 0.40],
        resize_lim=0.02,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=True,
        is_train=True,
    ),
    dict(
        type="BEVFusionGlobalRotScaleTrans",
<<<<<<< HEAD
        # scale_ratio_range=[0.95, 1.05],
        # rot_range=[-0.78539816, 0.78539816],
        # translation_std=[0.5, 0.5, 0.2],
        scale_ratio_range=[0.98, 1.02],
        rot_range=[-0.3925, 0.3925],
        translation_std=[0.2, 0.2, 0.1],
=======
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.2],
        # scale_ratio_range=[0.98, 1.02],
        # rot_range=[-0.3925, 0.3925],
        # translation_std=[0.2, 0.2, 0.1],
>>>>>>> e7daa8a9 (Added)
    ),
    dict(type="BEVFusionRandomFlip3D"),
    dict(type="ObjectRangeFilter", point_cloud_range=_base_.point_cloud_range),
    # Remove LiDAR points from the data
    dict(type="BEVFusionRemoveLiDARPoints"),
    dict(
        type="ObjectNameFilter",
        classes=[
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "transformation_3d_flow",
            "pcd_rotation",
            "pcd_scale_factor",
            "pcd_trans",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "timestamp",
            "vehicle_type",
            "city",
        ],
    ),
]

test_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        to_float32=True,
        color_type="color",
        backend_args=_base_.backend_args,
        camera_order=camera_order,
    ),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        # resize_lim=[0.34, 0.34],
        resize_lim=0.02,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False,
    ),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "num_pts_feats",
            "num_views",
            "timestamp",
            "vehicle_type",
            "city",
        ],
    ),
]

filter_cfg = dict(filter_frames_with_camera_order=camera_order)
