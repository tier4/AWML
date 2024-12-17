# Architecture for dataset pipeline
## The type of T4dataset

We divide the four types for T4dataset as following.

- Database T4dataset

Database T4dataset is mainly used for training a model.
We call database T4dataset as "Database {vehicle name}-vX.Y", "DB {vehicle name}-vX" in short.
We manage database T4dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `db_jpntaxi_v1.yaml` (file name use only the version of X).

- Use case T4dataset

Use case T4dataset is mainly used for evaluation with ROS environment.
We call Use case T4dataset as "Use case {vehicle name}-vX.Y", "UC {vehicle name}-vX" in short.

- Non-annotated T4dataset

Non-annotated T4dataset is the dataset which is not annotated.
After we annotate for it, it change to database T4dataset or use case T4dataset.

- Pseudo T4dataset

Pseudo T4dataset is annotated to non-annotated T4dataset by auto-labeling like [t4dataset_pseudo_label_3d](/tools/t4dataset_pseudo_label_3d/).
Pseudo T4dataset is mainly used to train pre-training model.
We call pseudo T4dataset as "Pseudo vX.Y.Z".
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

### Versioning strategy for database T4dataset

We manage the version of T4dataset as "the type of T4dataset" + "vehicle name" + "X.Y".
For example, we use like "DB jpntaxi-v2.2", "DB gsm8-v1.1", "Pseudo j6gen2-v1.0".

#### "the type of T4dataset"

As the type of T4dataset, we use "DB" for database T4dataset, "UC" for use case T4dataset, and "Pseudo" for pseudo T4dataset.

- "vehicle name".

For now, we define as following.

- jpntaxi: XX1 JPN TAXI
- gsm8: X2 Minibus (GSM8)
- j6: X2 Minibus 1.0 (J6)
- j6gen2: X2 Minibus 2.0 (J6)

- X: Management classification for dataset

It is recommended to change the number depending on the location and data set creation time.

- Y: The version of (dataset version X)

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

### Use case for update of database T4dataset

- We add new vehicle

We add database T4dataset for {new vehicle name}-v1.0.

- We add scenes (like different experiment area) for existing vehicle

We add database T4dataset from {vehicle name}-vX.Y to {vehicle name}-v(X+1).0.

- We add database T4dataset for the version {vehicle name}-vX.Y.

We update version {vehicle name}-vX.Y to version {vehicle name}-vX.(Y+1).

- In version {vehicle name}-vX.Y, we changed the trailer annotation method and requested the vendor to modify the annotations inside.

We create T4dataset entirely new since this case lead to destructive change for T4 format.
According to change of `T4 dataset ID`, we update from version {vehicle name}-vX.Y to version {vehicle name}-vX.(Y+1).

- 2D annotation did not exist in version X.Y.Z, so we add it.

We create T4dataset entirely new since this case lead to destructive change for T4 format.
According to change of `T4 dataset ID`, we update from version {vehicle name}-vX.Y to version {vehicle name}-vX.(Y+1).

- In version X.Y.Z, we found one vehicle that was not annotated, so we added and modified it by annotating it.

We update only `T4dataset WebAuto version` since this case is a non-destructive change for T4 format.
According to change of `T4dataset WebAuto version`, we update from version {vehicle name}-vX.Y to version {vehicle name}-vX.(Y+1).

- For pointcloud topic stored in rosbag of T4dataset, the data arrangement method was changed from XYZI to XYZIRC, and the contents of rosbag were also updated.

We update T4format and `T4dataset WebAuto version`.
According to update of these, we update from version {vehicle name}-vX.Y to version {vehicle name}-vX.Y.
