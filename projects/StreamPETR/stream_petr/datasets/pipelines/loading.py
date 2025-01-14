from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadAnnotations3D


@TRANSFORMS.register_module()
class StreamPETRLoadAnnotations3D(LoadAnnotations3D):
  """
    Overrides some of the methods to make the 
  """
  
  
  def _load_bboxes_depth(self, results):
    print(results['ann_info']['gt_bboxes_3d'])
    return super()._load_bboxes_depth(results)
  def _load_bboxes(self, results):
    print(results['ann_info']['gt_bboxes_3d'])
    return super()._load_bboxes(results)