from mmcv.transforms import BaseTransform
from mmdet3d.structures.ops import box_np_ops
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
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args