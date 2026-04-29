from mmengine.registry import DATASETS

from .t4dataset import T4Dataset


@DATASETS.register_module()
class T4DatasetFullCategories(T4Dataset):
    """T4Dataset variant that re-maps instance labels from gt_nusc_name at load time.

    The standard T4Dataset uses pre-computed bbox_label_3d stored in the pkl,
    which reflects a 5-class mapping. This subclass overrides label assignment
    so that an arbitrary broader class set can be used without regenerating the pkl.

    Args:
        extended_name_mapping (dict): Maps gt_nusc_name (as stored in pkl, i.e.
            the already-normalized annotation name such as "semi_trailer",
            "traffic_cone") to the new target class name. Names absent from this
            dict, or mapped to None, receive label -1 and are filtered out.
        class_names (list[str]): Ordered list of new class names. A gt_nusc_name
            that maps to a name not in this list also receives label -1.
        **kwargs: Forwarded to T4Dataset / NuScenesDataset.
    """

    def __init__(self, extended_name_mapping: dict, class_names: list, **kwargs):
        self._extended_name_mapping = extended_name_mapping
        self._ext_class_to_idx = {name: i for i, name in enumerate(class_names)}
        super().__init__(class_names=class_names, **kwargs)

    def _remap_instances(self, info: dict) -> dict:
        for instance in info.get("instances", []):
            raw = instance.get("gt_nusc_name", "")
            new_cls = self._extended_name_mapping.get(raw, None)
            new_label = self._ext_class_to_idx.get(new_cls, -1) if new_cls else -1
            instance["bbox_label"] = new_label
            instance["bbox_label_3d"] = new_label
        return info

    def parse_data_info(self, info: dict) -> dict:
        # Re-map labels before the parent class reads bbox_label_3d.
        info = self._remap_instances(info)
        return super().parse_data_info(info)
