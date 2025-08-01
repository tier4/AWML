_base_ = [
    "mmpretrain::_base_/default_runtime.py",
    "mmpretrain::_base_/models/resnet18.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
    "../../../../autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py",
]


batch_size = 8
num_workers = 8
max_epochs = 25

data_preprocessor = dict()

model = dict(
    type="mmpretrain.ImageClassifier",
    backbone=dict(
        type="ResNet",
        depth=18,
        in_channels=5,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

# Overwrite default optimization settings if needed
optim_wrapper = dict(optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(type="MultiStepLR", by_epoch=True, milestones=[5, 15, 20], gamma=0.1)


train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)


# User setting
# Dataset setting
data_root = "/workspace/data/t4dataset"
info_directory_path = "/workspace/data/t4dataset/calibration_info/"

# Visualization setting
train_projection_vis_dir = None

val_projection_vis_dir = None
val_results_vis_dir = None
test_projection_vis_dir = "./test_projection_vis_t4dataset/"
test_results_vis_dir = "./test_results_vis_t4dataset/"


train_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="train",
        undistort=True,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=train_projection_vis_dir,
        results_vis_dir=None,
    ),
    dict(
        type="PackInputs",
        input_key="fused_img",
    ),
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + _base_.info_train_file_name,
        pipeline=train_pipeline,
        data_root=data_root,
    ),
)

val_cfg = dict()

val_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="val",
        undistort=True,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=val_projection_vis_dir,
        results_vis_dir=val_results_vis_dir,
    ),
    dict(
        type="PackInputs",
        input_key="fused_img",
        # Uncomment to use meta_keys for result visualization
        # meta_keys=["img_path", "fused_img", "image", "sample_idx", "frame_id", "frame_idx"],
    ),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=val_pipeline,
        data_root=data_root,
    ),
)

val_evaluator = dict(topk=(1,), type="mmpretrain.evaluation.Accuracy")

test_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="val",
        undistort=True,
        enable_augmentation=False,
        data_root=data_root,
        projection_vis_dir=test_projection_vis_dir,
        results_vis_dir=test_results_vis_dir,
    ),
    dict(
        type="PackInputs",
        input_key="fused_img",
        meta_keys=["img_path", "fused_img", "image", "sample_idx", "frame_id", "frame_idx"],
    ),
]

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=test_pipeline,
        data_root=data_root,
        # indices=5,  # Use first 5 samples for testing
    ),
)

test_evaluator = val_evaluator


custom_hooks = []
# Add visualization hooks for val and test results if their respective directories are specified
# Note: Training phase only supports projection visualization, not result visualization
if val_results_vis_dir is not None:
    val_hook_config = dict(type="ResultVisualizationHook", results_vis_dir=val_results_vis_dir, phases=["val"])
    if data_root is not None:
        val_hook_config["data_root"] = data_root
    custom_hooks.append(val_hook_config)

if test_results_vis_dir is not None:
    test_hook_config = dict(type="ResultVisualizationHook", results_vis_dir=test_results_vis_dir, phases=["test"])
    if data_root is not None:
        test_hook_config["data_root"] = data_root
    custom_hooks.append(test_hook_config)


vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)
