# Model Rareness Example Mining
A scene selection tool designed to identify scenes containing rare or uncertain objects based on predictions from multiple object detection models. It leverages the uncertainty estimation provided by the ModelRareExampleMining method to make decisions about whether a scene should be flagged as a "target scene."

# Note
For bevusion make sure to change the config to include nms_type='rotated'
```python
        test_cfg=dict(
            ...
            nms_type="rotated",
            pre_max_size=None,
            post_max_size =None,
            ...
        )
```
, and change transfusion config to include nms_type='circle'.
```python
        test_cfg=dict(
            ...
            nms_type="circle",
            ...
        )
```
