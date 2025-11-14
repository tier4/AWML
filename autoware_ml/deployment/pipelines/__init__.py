"""
Deployment Pipelines for Complex Models.
This module provides pipeline abstractions for models that require
multi-stage processing with mixed PyTorch and optimized backend inference.
"""

# # CenterPoint pipelines (3D detection)
# from .centerpoint import (
#     CenterPointDeploymentPipeline,
#     CenterPointPyTorchPipeline,
#     CenterPointONNXPipeline,
#     CenterPointTensorRTPipeline,
# )

# # YOLOX pipelines (2D detection)
# from .yolox import (
#     YOLOXDeploymentPipeline,
#     YOLOXPyTorchPipeline,
#     YOLOXONNXPipeline,
#     YOLOXTensorRTPipeline,
# )

# # Calibration pipelines (classification)
# from .calibration import (
#     CalibrationDeploymentPipeline,
#     CalibrationPyTorchPipeline,
#     CalibrationONNXPipeline,
#     CalibrationTensorRTPipeline,
# )

# __all__ = [
#     # CenterPoint
#     'CenterPointDeploymentPipeline',
#     'CenterPointPyTorchPipeline',
#     'CenterPointONNXPipeline',
#     'CenterPointTensorRTPipeline',
#     # YOLOX
#     'YOLOXDeploymentPipeline',
#     'YOLOXPyTorchPipeline',
#     'YOLOXONNXPipeline',
#     'YOLOXTensorRTPipeline',
#     # Calibration
#     'CalibrationDeploymentPipeline',
#     'CalibrationPyTorchPipeline',
#     'CalibrationONNXPipeline',
#     'CalibrationTensorRTPipeline',
# ]
