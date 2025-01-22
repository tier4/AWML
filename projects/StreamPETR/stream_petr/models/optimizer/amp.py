# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
import torch.nn as nn

from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

@OPTIM_WRAPPERS.register_module()
class NoCacheAmpOptimWrapper(AmpOptimWrapper):
    """
      The gradients disappear for the Linear layers when using mixed precision training, so need to disable the cache.
    """

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        from mmengine.runner.amp import autocast
        with super().optim_context(model), autocast(dtype=self.cast_dtype, cache_enabled=False):
            yield