# Architecture for dataset pipeline
## T4dataset

T4dataset is the dataset format we define.
In detail, please see [T4dataset document](https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md).
The type of T4dataset is following

- Database dataset

It is used for training and evaluation.
We manage database dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `database_v1_0.yaml`.

- Pseudo dataset

It is mainly used to train pre-training model.
We manage database dataset in [dataset config](/autoware_ml/configs/detection3d/dataset/t4dataset) like `pseudo_v1_0.yaml`.
Pseudo T4dataset is created by [t4dataset_pseudo_label_3d](/tools/t4dataset_pseudo_label_3d/).
Note that `autoware-ml` do not manage pseudo T4dataset which is used for domain adaptation.

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
