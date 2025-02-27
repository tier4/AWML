# Copyright (c) OpenMMLab. All rights reserved.
# Copied from https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/models/layers/spconv/__init__.py
# NOTE(knzo25): needed to overwrite our custom deployment oriented operations

from .overwrite_spconv import register_spconv2

try:
    import spconv
except ImportError:
    IS_SPCONV2_AVAILABLE = False
else:
    if hasattr(spconv, "__version__") and spconv.__version__ >= "2.0.0":
        IS_SPCONV2_AVAILABLE = register_spconv2()
    else:
        IS_SPCONV2_AVAILABLE = False
