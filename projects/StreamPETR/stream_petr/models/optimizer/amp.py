# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper, OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class NoCacheAmpOptimWrapper(AmpOptimWrapper):
    """
    The gradients disappear for the Linear layers when using mixed precision training, so need to disable the cache.
    This happens because there are no_grad operations used for a few forward passes before using forward passes with grad
    https://github.com/pytorch/pytorch/issues/142234
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

    def backward(self, loss: torch.Tensor, **kwargs):
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """
        # with torch.autograd.detect_anomaly():
        self.loss_scaler.scale(loss).backward(**kwargs)
        self._inner_count += 1


@OPTIM_WRAPPERS.register_module()
class DebugOptimWrapper(OptimWrapper):

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        with torch.autograd.detect_anomaly():
            loss.backward(**kwargs)
        self._inner_count += 1
