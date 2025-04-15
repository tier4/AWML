# PointcloudSegmentationSAM2

This projects leverages SAM2 and grounding dino to segment pointclouds via intermediate image segmentation and pointcloud projection.
The process can be divided in two steps:

 - Image segmentation: Using SAM2 and grounding dino, images of the dabases are segmented
   attempting to assign a label to all parts of the image
 - Pointcloud segmentation via projection: Labels are assigned to points via projecting
   them to all the cameras in the rig, using several time steps for consistency.

This projects is applied to `t4dataset`, but can be easily generalized.
`info` files are not required and the output of the pointcloud segmentation is not added into the `infos` for now.
Instead, image and pointclouds segmentation results are saved alongside the dataset data.

## Installation

Note: due to SAM2 requirements, the image from this projects uses a different version of torch and other dependencies.

Build the image:

```bash
DOCKER_BUILDKIT=1 docker build -t autoware-ml-sam2 -f projects/SAM2/Dockerfile . --progress=plain
```

To execute the container:

```bash
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml-sam2
```

Before following to the next steps, the checkpoints for SAM2 and grounding dino need to be downloaded.

```bash
TODO
```

## Generate SAM2 segmented images

To segment the images of a dataset using SAM2, use:

```bash
python projects/SAM2/segment_t4dataset_sam2.py \
    --root_path ./data/t4dataset \
    --out_videos ./videos \
    --dataset_config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py \
    --sam2_config projects/SAM2/config/t4dataset_segment.yaml \
    --override 1 \
    --only_key_frames 0
```

The `segmentation_config` specifies the specifics of `SAM2` including the specific model, checkpoints, and thresholds.
The classes queries are aimed to specify as many as possible of the elements in a scene, since only non-background
objects will be assigned valid labels in the pointclouds in the next step.

Segmented images will be generated alongside the original images with the `_seg.png` suffix.

## Generate segmented pointclouds using projection

To generate segmented pointclouds use the following command:

```bash
python projects/SAM2/segment_t4dataset_projective.py \
    --root_path ./data/t4dataset \
    --database_config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py \
    --segmentation_config projects/SAM2/config/t4dataset_segment.yaml
```

The classes that are used for pointcloud segmentation are a subset of the ones used in SAM2. This is due to the
need for relatively need specifity in SAM2 queries to obtain a good sensitivity.

For example, `greenery`, `bush`, and `trees` are all classes that we would like to classify as `vegetation` in a pointcloud.
However, `SAM2` does not have a high sensitivity towards `vegetation`, for which reason in the  previous step, many
syonyms and related terms are required as queries.

This script will project the pointcloud into the camera rig of the instance associated with the lidar, and those before
and after it, controlled via `num_consistent_frames`.

The rules for segmentation as as follow:
 - Points that are not projected in any image are classified as invalid
 - Points that are projected only into images at pixels classified as background, the points will be classified as invalid.
 - Points that are projected into images at pixels with different non background classes, the points will be classified as invalid.
 - Points that are projected into at least one image to a non background class, and all the classes coincide, the points will be
   classified as the projected class.
 - The borders between classes in the segmented image are considered as background (morphological dilation in the contour)

In this context, invalid means that the label is unknown, and should be masked out during pointcloud segmentation training.

Limitation of the projective approach:
 - Due to sensor calibration, vehicle movement, and lidar scanning, classification will leak between objects. This is
   somewhat addressed through temporal consistency and morphological operations.
 - Projective approaches, due to the baseline and the nature of the sensors, will provide wrong labels in some cases,
   even when the image segmentation is perfect (e.g., vehicle behind a fence).
 - Some parts of the pointcloud will not be classified and will be masked out during pointcloud classification training.
   If this phenomenah is consistent, some objects will never receive labels and have potential errors at test time.

## (Optional) Refine pointcloud segmentation with object detection cuboids

It is possible to refine the segmentation labels using the cuboids from a object detection groundtruth.
For the `t4dataset`, it can be done using the following command:

```bash
python projects/SAM2/segment_t4dataset_projective.py \
    --root_path ./data/t4dataset \
    --database_config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py \
    --segmentation_config projects/SAM2/config/t4dataset_segment.yaml
```

## (Optional) Generate BEV videos with the segmentation result

BEV videos of the setmented pointclouds can be generated with the following command:

```bash
python projects/SAM2/generate_segmentation_videos.py \
    --root_path ./data/t4dataset \
    --out_videos ./videos \
    --dataset_config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py \
    --segmentation_config projects/SAM2/config/t4dataset_segment.yaml
```
