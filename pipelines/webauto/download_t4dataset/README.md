# Download T4dataset by WebAuto CLI

You can download T4dataset by script using WebAuto CLI.

## Environment

- This script do not need docker environment and is tested by Ubuntu22.04 LTS environment.
- You check data access right of [WebAuto(> v0.33.1)](https://docs.web.auto/en/user-manuals/).

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## Get started

- (choice) Download for XX1

```sh
python tools/setting_environment/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v1_1.yaml --output ./data/t4dataset/ --project-id prd_jt
```

- (choice) Download for X2

```sh
python tools/setting_environment/download_t4dataset.py autoware_ml/configs/detection3d/dataset/t4dataset/database_v3_0.yaml --output ./data/t4dataset/ --project-id x2_dev
```

- After download as above command, the directory architecture consists as below.

```
- data/t4dataset
  - /database_v1_1/
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
    - {t4dataset_id}
```
