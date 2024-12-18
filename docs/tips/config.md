# Docs for configs
## Tips
### How to confirm the parameter

1. Confirm the model config

Please confirm the model you use.
The model is in `/projects/*` and confirm what configs you should use.
As an example, we use [CenterPoint](/projects/CenterPoint/) and [this config file](https://github.com/tier4/autoware-ml/tree/e73f827483d49af53fe0aa4f1e7aebccf720971a/projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_x2.py).

2. Confirm the parameter to train

To confirm the parameters for training, please see like [the parameters of config file](https://github.com/tier4/autoware-ml/tree/e73f827483d49af53fe0aa4f1e7aebccf720971a/projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_x2.py#L305).

3. Confirm the dataset config

From [the import file](https://github.com/tier4/autoware-ml/tree/e73f827483d49af53fe0aa4f1e7aebccf720971a/projects/CenterPoint/configs/t4dataset/second_secfpn_2xb8_121m_x2.py#L3), confirm the dataset config in `/autoware_ml/configs/detection3d/dataset/`.
As an example, please confirm like [dataset config](https://github.com/tier4/autoware-ml/tree/e73f827483d49af53fe0aa4f1e7aebccf720971a/autoware_ml/configs/detection3d/dataset/t4dataset/x2.py#L14).
An [the yaml files](https://github.com/tier4/autoware-ml/tree/e73f827483d49af53fe0aa4f1e7aebccf720971a/autoware_ml/configs/detection3d/dataset/t4dataset/db_gsm8_v1.yaml) are defined and it has the information of dataset.

### How to manage the config files

The config file of [MMlab](https://github.com/open-mmlab) libraries has high degree of freedom to set parameters.
However, it is easy to lead technical debt because config files are scattered.

So we recommend following config directory structure.

```
- autoware_ml/
  - configs/
    - detection3d/
      - component.py
- projects/
  - TransFusion/
    - configs/
      - t4dataset/
        - 90m_768grid/
          - default.py
          - parameter_1.py
          - parameter_2.py
        - 50m_768grid/
          - default.py
```

## Detection3d
### Use for Tensorboard

- Add backend for Tensorboard to config

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
```
