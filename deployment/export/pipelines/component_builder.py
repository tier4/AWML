"""
Model component builder: the seam that splits a model into exportable parts.

``ExportableComponent`` is the unit exchanged across the seam (one ONNX file
per component); ``ModelComponentBuilder`` is the contract the ONNX export
pipeline depends on to turn a model + sample into those components.
``DefaultComponentBuilder`` is the built-in implementation for the common
whole-model case (one component = the whole model); projects whose model must
be decomposed (e.g. CenterPoint) provide their own builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import torch

from deployment.config.schema import ComponentsConfig

if TYPE_CHECKING:
    import onnx


@dataclass(frozen=True)
class ExportableComponent:
    """A model component ready for ONNX export.

    Attributes:
        name: Component identifier (same as key in deploy config components). Used for
              config lookup, output filename, and logs.
        module: PyTorch module to export.
        sample_input: Sample input tensor for tracing.
        post_transforms: Optional ONNX graph transforms applied (in order) to the exported
            file after ``torch.onnx.export``. Each takes the loaded ``onnx.ModelProto`` and
            returns the (possibly mutated) model. Empty for models needing no post-processing
            (e.g. CenterPoint); used by BEVFusion for the TopK-constant fix and ImplicitGemm+ReLU
            fusion.
    """

    name: str
    module: torch.nn.Module
    sample_input: Any
    post_transforms: Tuple[Callable[["onnx.ModelProto"], "onnx.ModelProto"], ...] = ()


class ModelComponentBuilder(ABC):
    """Interface for building exportable ONNX components from model and sample."""

    @abstractmethod
    def build_components(
        self,
        model: torch.nn.Module,
        sample: Any,
    ) -> List[ExportableComponent]:
        """Build all ONNX-exportable components.

        Args:
            model: PyTorch model to build components from
            sample: Typed sample payload for preparing component inputs

        Returns:
            List of exportable model components ready for ONNX export.
        """
        ...


class DefaultComponentBuilder(ModelComponentBuilder):
    """Builder that exports the whole model as a single ONNX component.

    The deploy config must declare exactly one component; the whole model maps
    onto it. A config with several components but no decomposition logic is a
    misconfiguration (it would otherwise export identical copies of the full
    model), so it is rejected with an explicit error.
    """

    def __init__(self, components_cfg: ComponentsConfig) -> None:
        """Initialize the builder.

        Args:
            components_cfg: Component config; must declare exactly one component.
        """
        self._components_cfg = components_cfg

    def build_components(self, model: torch.nn.Module, sample: Any) -> List[ExportableComponent]:
        """Return a single exportable component wrapping the whole model.

        Args:
            model: PyTorch model exported as one component.
            sample: Preprocessed tracing input from the sample extractor.

        Returns:
            A one-element list with the whole model as the component.

        Raises:
            ValueError: If the deploy config declares other than one component.
        """
        names = list(self._components_cfg.component_names())
        if len(names) != 1:
            raise ValueError(
                f"DefaultComponentBuilder maps the whole model onto exactly one component, but the "
                f"deploy config declares {len(names)}: {names}. A model that must be split into "
                f"multiple ONNX components needs a project-specific ModelComponentBuilder."
            )
        return [ExportableComponent(name=names[0], module=model, sample_input=sample)]
