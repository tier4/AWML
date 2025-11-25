import torch
import torch.nn.functional as F
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ResizeSegMask:
    def __init__(self, size):
        self.size = size  # (H_out, W_out)

    def __call__(self, results):
        if 'gt_seg_map' in results and results['gt_seg_map'] is not None:
            seg = results['gt_seg_map']  # numpy array (H, W)
            seg = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
            seg = F.interpolate(seg, size=self.size, mode='nearest')
            results['gt_seg_map'] = seg.squeeze(0).squeeze(0).long().numpy()  # back to (H_out,W_out)
        return results