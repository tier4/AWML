### Interface

Divide the interface for `AWML` and dataset pipelines (internal use)

- 1. Algorithm for scene selector

We develop the algorithm for scene selector in `AWML`.
For example, to make the selector of "a traffic cone placed on the road near an intersection", we will make the combination with ntersection determination using VLM, roadway detection using semantic segmentation, and traffic cone detection using 2D detection.

By dividing the interface and making the script to test, we can develop in `AWML` as prototype development of the algorithm.
After prototyping, we move to our dataset pipelines.

- 2. Integration for dataset pipelines with DevOps

To apply data mining for large amount of rosbag, we integrate the selector to our dataset pipelines.

### Policy

We set policy to develop scene selector.
We develop

1. Scene selector class

We use scene selector class because we make it easier to handle and operate.

```py
class MySceneSelector(SceneSelector):

    def __init__() -> None:
        pass

    def is_target_scene(image_array: list[np.ndarray]) -> bool:
        pass
```

When we follow this policy, we can move the algorithm scene selector between each repository.

2. Use config file

We use config files based on MMLab series.
It can follow the whole architecture in `AWML`.
