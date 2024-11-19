from mmdet3d.models import CenterPoint as _CenterPoint
from mmdet3d.registry import MODELS


@MODELS.register_module(force=True)
class CenterPoint(_CenterPoint):
    """
    CenterPoint based on mmdet3d.models.CenterPoint.
    Create another module for possible customization in the future.
    """

    def __init__(self, **kwargs):
        """
        Initialization of CenterPoint.
        """
        super(CenterPoint, self).__init__(**kwargs)
