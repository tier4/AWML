# Architecture for dataset pipeline
## The type of T4dataset

We divide the four types for T4dataset as following.

- Database T4dataset

Database T4dataset is mainly used for training a model.
We call database T4dataset as "Database {vehicle name} vX.Y", "DB {vehicle name} vX.Y" in short.
For example, we use like "Database JPNTAXI v1.1", "DB JPNTAXI v1.1" in short.
We manage database T4dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `db_jpntaxi_v1.yaml` (file name use only the version of X).

- Use case T4dataset

Use case T4dataset is mainly used for evaluation with ROS environment.
We call Use case T4dataset as "Use case {vehicle name} vX.Y", "UC {vehicle name} vX.Y" in short.

- Non-annotated T4dataset

Non-annotated T4dataset is the dataset which is not annotated.
After we annotate for it, it change to database T4dataset or use case T4dataset.

- Pseudo T4dataset

Pseudo T4dataset is annotated to non-annotated T4dataset by auto-labeling like [t4dataset_pseudo_label_3d](/tools/t4dataset_pseudo_label_3d/).
Pseudo T4dataset is mainly used to train pre-training model.
We call pseudo T4dataset as "Pseudo {vehicle name} vX.Y".
For example, we use like "Pseudo JPNTAXI v1.0".
We manage pseudo T4dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `pseudo_jpntaxi_v1.yaml` (file name use only the version of X).
Note that `autoware-ml` do not manage Pseudo T4dataset which is used for domain adaptation.

## T4dataset

We define T4dataset, which is based on nuScenes format.
The directory architecture is following.

```
- dataset_directory/
  - {The type of T4dataset + dataset version}
    - {T4 dataset ID}/
      - {T4dataset WebAuto version}
        - annotation/
        - data/
        - input_bag/
        - map/
    - ...
```

### Versioning strategy for Database T4dataset

We manage the version of T4dataset as "the type of T4dataset" + "vehicle name" + "vX.Y".
For example, we use like "DB JPNTAXI v2.2", "DB GSM8 v1.1", "Pseudo J6Gen2 v1.0".

#### "the type of T4dataset"

As the type of T4dataset, we use "DB" for database T4dataset, "UC" for use case T4dataset, and "Pseudo" for pseudo T4dataset.

#### "vehicle name".

We define as following.

- "JPNTAXI"

"JPNTAXI" is the taxi based on "JPN TAXI".
It is categorized in Robo-Taxi, what we call "XX1".
The sensor configuration basically consists of 1 * VLS-128 as top LiDAR, 3 * VLP-16 as side LiDARs, 6 * TIER IV C1 cameras (85deg) as cameras for object recognition, 1 * TIER IV C1 camera (85deg) as camera for traffic light recognition, 1 * TIER IV C2 camera (62deg) as camera for traffic light recognition, and 6 * continental ARS408-21 as radars.
The sensors depends on when the data was taken.

- "GSM8"

"GSM8" is the EV mini-bus version 0 based on the vehicle of "GSM8"
It is categorized in shuttle bus, what we call "X2".
The sensor configuration consists of 4 * Hesai Pandar40P as main LiDARs, 4 * Hesai PandarQT64 as LiDARs for surround objects, 6 * TIER IV C1 cameras (85deg) as cameras for object recognition, 2 * TIER IV C1 camera (85deg) as camera for traffic light recognition.

- "J6"

"J6" is the EV mini-bus version 1 based on the vehicle of "J6"
It is categorized in shuttle bus, what we call "X2".
The sensor configuration is basically same as "GSM8".
The sensor configuration consists of 4 * Hesai Pandar40P as main LiDARs, 4 * Hesai PandarQT64 as LiDARs for surround objects, 6 * TIER IV C1 cameras (85deg) as cameras for object recognition, 2 * TIER IV C1 camera (85deg) as camera for traffic light recognition, and 6 * continental ARS408 as radars.

- "J6Gen2"

"J6Gen2" is the EV mini-bus version 2 based on the vehicle of "J6" and update the sensor configuration from "J6".
It is categorized in gen2 of shuttle bus, what we call "X2Gen2".
The sensor configuration basically consists of 4 * Hesai OT128 as main LiDARs, 4 * Hesai QT128 as LiDARs for surround objects, 9 * TIER IV C1 cameras (85deg) as cameras for object recognition, 1 * TIER IV C2 camera (62deg) as camera for traffic light recognition, 1 * TIER IV C2 camera (30deg) as camera for traffic light recognition, and 6 * continental ARS540 as radars.

- "TLR"

In addition to T4dataset with each vehicle, we construct T4dataset for traffic light recognition (TLR).
It don't depends on the type of vehicles, so we call the dataset as "TLR".

#### X: Management classification for dataset

It is recommended to change the number depending on the location and data set creation time.

#### Y: The version of dataset

Upgrade the version every time a change may have a negative impact on performance for training.
For example, if we change of the way to annotation, we update the dataset and this version.
If we add new dataset, we update this version.
If we update the T4 format, we update the dataset and this version.

### T4 dataset ID

`T4 dataset ID` is the id managed in [WebAuto system](https://docs.web.auto/en/) as `70891309-ca8b-477b-905a-5156ffb3df65`.

### T4dataset WebAuto version

`T4dataset WebAuto version` is the version of T4dataset itself.
If we fix annotation or sensor data, we update this version.
When we make a T4dataset, we start from version 0.

### T4format

We define T4format, which defines detail schema for T4dataset.
We manage the version of T4format.
If you want to know about detailed schema and the version of T4format, please see [document of T4format](https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md).

## Whole data pipeline for ML model with Autoware
### Pipeline

![](/docs/fig/data_pipeline.drawio.svg)

- 1. Make non-annotated T4dataset

Create non-annotated T4dataset from rosbag file by using [T4dataset tools](https://github.com/tier4/tier4_perception_dataset).

- 2. Semi-auto labeling

`autoware-ml` make pseudo-annotated T4dataset from non-annotated T4dataset.
It is used for training with pseudo label and semi-auto labeling to make T4dataset.
Semi-auto labeling makes short time to human annotation.
In addition to it, pseudo-annotated T4dataset is also used for domain adaptation and training of pretrain model.

- 3. Human annotation

From pseudo-annotated T4dataset or non-annotated T4dataset, dataset tools convert to the format of each annotation tools.
Annotated dataset is made by human annotation and then annotated T4dataset is created.

- 4. Dataset management

We upload to [WebAuto](https://web.auto/) system and manage T4dataset.
`autoware-ml` use T4dataset downloading by WebAutoCLI.

### Update T4dataset

Please see [contribution_use_case](/docs/contribution/contribution_use_case.md)
