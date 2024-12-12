from typing import List, Union

import numpy as np

from .scene_selector import SceneSelector


class ImageBasedSceneSelector(SceneSelector):

    def __init__() -> None:
        pass

    def is_target_scene(
        self,
        image_array: Union[List[np.ndarray], List[str]],
        *args,
        **kwargs,
    ) -> bool:
        pass

    def is_target_scene_multiple(
        self, multiple_image_arrays: List[Union[List[np.ndarray], List[str]]], *args, **kwargs
    ) -> List[bool]:
        pass
