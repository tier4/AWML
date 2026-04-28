from mmcv.transforms import BaseTransform
from mmdet3d.structures.ops import box_np_ops
from mmdet3d.datasets.transforms import LoadPointsFromFile, LoadPointsFromMultiSweeps
from mmengine.registry import TRANSFORMS



@TRANSFORMS.register_module()
class LoadPointsFromCurrentFileSweep(BaseTransform):
    """Load points from the current file and sweep. 
    This is used to load the points from the current file and sweep for copy-paste augmentation.

    Args:
        coord_type (str): The type of coordinates of points cloud.
        load_dim (int): The dimension of the loaded points.
        use_dim (list[int] | int): Which dimensions of the points to use.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None, 
                 sweeps_num: int = 10,
                 pad_empty_sweeps: bool = False,
                 remove_close: bool = False,
                 test_mode: bool = False
                 ) -> None:
        
        self.points_loader = LoadPointsFromFile(coord_type=coord_type, load_dim=load_dim, use_dim=use_dim, backend_args=backend_args)
        if sweeps_num > 0:
            self.points_from_multi_sweeps_loader = LoadPointsFromMultiSweeps(sweeps_num=sweeps_num, pad_empty_sweeps=pad_empty_sweeps, remove_close=remove_close, test_mode=test_mode)
        else:
            self.points_from_multi_sweeps_loader = None

    def transform(self, results: dict) -> dict:
        points = self.points_loader(results)
        if self.points_from_multi_sweeps_loader is not None:
            points = self.points_from_multi_sweeps_loader(points)
        return points
