import numpy as np

from tools.scene_selector.scene_selector.image_based_scene_selector import \
    ImageBasedSceneSelector

class Det2dObjectNumSelector(ImageBasedSceneSelector):
    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        confidence_threshold: list[float],
        target_label: list[str],
        threshold_object_num: int,
    ) -> None:
        pass

    def is_target_scene(
        self,
        image_array: list[np.array],
    ) -> bool:
        for image in image_array:
            output = self.model(image)
            if len(output.bbox) > self.object_num_threshold:
                return True
            return False
