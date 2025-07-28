_base_ = [
    "mmpretrain::_base_/default_runtime.py",
    "mmpretrain::_base_/models/resnet18.py",
    "mmpretrain::_base_/schedules/imagenet_bs256.py",
]

custom_imports = dict(
    imports=["autoware_ml.classification2d.datasets.transforms.calibration_classification_transform"],
    allow_failed_imports=False,
)

batch_size = 8
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

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

train_pipeline = [
    dict(type="CalibrationClassificationTransform"),
    dict(type="PackInputs", input_key="img"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    shuffle=True,
    dataset=dict(
        type="mmpretrain.CustomDataset",
        data_prefix="data/calibrated_data/training_set",
        with_label=True,
        pipeline=train_pipeline,
    ),
)

val_cfg = dict()

val_pipeline = [
    dict(type="CalibrationClassificationTransform", validation=True),
    dict(type="PackInputs", input_key="img"),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size,
    persistent_workers=False,
    shuffle=False,
    dataset=dict(
        type="mmpretrain.CustomDataset",
        data_prefix="data/calibrated_data/validation_set",
        with_label=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(topk=(1,), type="mmpretrain.evaluation.Accuracy")

test_pipeline = [
    dict(type="CalibrationClassificationTransform", test=True),
    dict(type="PackInputs", input_key="img"),
]

test_dataloader = val_dataloader

test_evaluator = val_evaluator

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)
