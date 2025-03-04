# Make new library

There are many cases that you want to make as new library with `AWML`.
Here is examples.

- An engineer wants to use `AWML` for a secret project with private own dataset
- An engineer wants to construct ML library for new task with `AWML`

If you face these cases, we recommend the procedure as below.

## 1. Make new repository

```
- your_new_repository/
```

## 2. Use `AWML` as submodule

- We recommend to fix the version of `AWML`

```
- your_new_repository/
  - AWML (submodule)
```

- Here is commands to add submodule.

```
cd your_new_repository
git submodule add https://github.com/tier4/AWML AWML
git commit -m "add AWML as a submodule"
```

## 3. Add feature
### 3.1 Add new model

- If you want to a new model, you can add to own `projects/`

```
- your_new_repository/
  - AWML (submodule)
  - projects/
    - your_new_algorithm/
```

### 3.2 Add new dataset

- If you want to new dataset, you should add new config files to library directory and set sensor configs like `AWML`.

```
- your_new_repository/
  - AWML (submodule)
  - {your_package_name}/
    - configs/
      - your_dataset/
        - your_dataset.yaml
      - detection3d/
        - dataset/
          - t4dataset/
            - base.py
```

- If you want to use from model, you add new config file for projects

```
- your_new_repository/
  - AWML (submodule)
  - {your_package_name}/
    - configs/
      - your_dataset/
        - your_dataset.yaml
      - detection3d/
        - dataset/
          - t4dataset/
            - base.py
  - projects/
    - CenterPoint/
      - configs/
        - new_config.py
```

- new_config.py
  - Note that the the hierarchy is off by one level from `AWML`.

```py
_base_ = [
    #"../../../../autoware_ml/configs/detection3d/default_runtime.py",
    #"../../../../autoware_ml/configs/detection3d/dataset/t4dataset/base.py",
    "../../../../AWML/autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../AWML/autoware_ml/configs/detection3d/dataset/t4dataset/base.py",
    "../models/centerpoint_second_secfpn_base.py",
]
```

### 3. Add library

- If you want to add new library, you can add library on your own repository.

```
- your_new_repository/
  - AWML (submodule)
  - {your_package_name}/
    - new_library.py
```
