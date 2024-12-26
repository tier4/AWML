# Download T4dataset by WebAuto CLI

You can download T4dataset by script using WebAuto CLI.

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## Environment

- This script do not need docker environment and is tested by Ubuntu22.04 LTS environment.
- Download WebAuto CLI and setup as [the document](https://docs.web.auto/en/developers-guides/quick-start)
- You check data access right of [WebAuto(> v0.33.1)](https://docs.web.auto/en/user-manuals/).

## Get started

- (Choice) Download for XX1
  - If you want to use rosbag, delete the flag of `--delete-rosbag`

```sh
python pipelines/webauto/download_t4dataset/download_t4dataset.py autoware_ml/configs/t4dataset/db_jpntaxi_v2.yaml --output ./data/t4dataset/ --project-id prd_jt --delete-rosbag
```

- (Choice) Download for X2

```sh
python pipelines/webauto/download_t4dataset/download_t4dataset.py autoware_ml/configs/t4dataset/db_j6_v1.yaml --output ./data/t4dataset/ --project-id x2_dev --delete-rosbag
```

- (Choice) Download for traffic light recognition

```sh
python pipelines/webauto/download_t4dataset/download_t4dataset.py autoware_ml/configs/t4dataset/db_tlr_v1.yaml --output ./data/t4dataset/ --project-id prd_jt --delete-rosbag
```

- After download as above command, the directory architecture consists as below.

```
- data/t4dataset
  - /db_jpntaxi_v2/
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
```
