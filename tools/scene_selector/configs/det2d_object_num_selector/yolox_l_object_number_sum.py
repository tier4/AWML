t4_dataset_sensor_names = ['CAM_FRONT','CAM_BACK']
batch_size = 8

classes = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

scene_selector = dict(
    type="Det2dObjectNumSelector",
    model_config_path=
    "/workspace/projects/YOLOX/configs/yolox_l_8xb8-300e_coco.py",
    model_checkpoint_path=
    "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    confidence_threshold=0.5,
    classes=classes,
    target_and_threshold={
        'person': 10,
        'bicycle': 2,
        'car': 10,
        'motorcycle': 2,
        'bus': 5,
        'train': 1,
        'truck': 5,
        'boat': 1,
        'fire hydrant': 1,
        'stop sign': 1,
        'parking meter': 1,
        'bird': 5,
        'cat': 1,
        'dog': 1,
        'umbrella': 1,
        'sports ball': 1,
        'skateboard': 1,
    },
    batch_size=batch_size,
)
