"""
Preprocessing pipeline builder for deployment data loaders.

This module provides functions to extract and build preprocessing pipelines
from MMDet/MMDet3D/MMPretrain configs for use in deployment data loaders.

This module is compatible with the BaseDeploymentPipeline.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope

logger = logging.getLogger(__name__)

TransformConfig = Mapping[str, Any]


class ComposeBuilder:
    """
    Unified builder for creating Compose objects with different MM frameworks.

    Uses MMEngine-based Compose with init_default_scope for all frameworks.
    """

    @staticmethod
    def build(
        pipeline_cfg: List[TransformConfig],
        scope: str,
        import_modules: List[str],
    ) -> Any:
        """
        Build Compose object using MMEngine with init_default_scope.

        Args:
            pipeline_cfg: List of transform configurations
            scope: Default scope name (e.g., 'mmdet', 'mmdet3d', 'mmpretrain')
            import_modules: List of module paths to import for transform registration

        Returns:
            Compose object

        Raises:
            ImportError: If required packages are not available
        """
        # Import transform modules to register transforms
        for module_path in import_modules:
            try:
                __import__(module_path)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import transform module '{module_path}' for scope '{scope}'. "
                    f"Please ensure the required package is installed. Error: {e}"
                ) from e

        # Set default scope and build Compose
        try:
            init_default_scope(scope)
            logger.info(
                "Building pipeline with mmengine.dataset.Compose (default_scope='%s')",
                scope,
            )
            return Compose(pipeline_cfg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build Compose pipeline for scope '{scope}'. "
                f"Check your pipeline configuration and transforms. Error: {e}"
            ) from e


TASK_PIPELINE_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "detection2d": {
        "scope": "mmdet",
        "import_modules": ["mmdet.datasets.transforms"],
    },
    "detection3d": {
        "scope": "mmdet3d",
        "import_modules": ["mmdet3d.datasets.transforms"],
    },
    "classification": {
        "scope": "mmpretrain",
        "import_modules": ["mmpretrain.datasets.transforms"],
    },
    "segmentation": {
        "scope": "mmseg",
        "import_modules": ["mmseg.datasets.transforms"],
    },
}

# Valid task types
VALID_TASK_TYPES = list(TASK_PIPELINE_CONFIGS.keys())


def build_preprocessing_pipeline(
    model_cfg: Config,
    task_type: str = "detection3d",
) -> Any:
    """
    Build preprocessing pipeline from model config.

    This function extracts the test pipeline configuration from a model config
    and builds a Compose pipeline that can be used for preprocessing in deployment data loaders.

    Args:
        model_cfg: Model configuration containing test pipeline definition.
                   Supports config (``model_cfg.test_pipeline``)
        task_type: Explicit task type ('detection2d', 'detection3d', 'classification', 'segmentation').
                   Must be provided either via this argument or via
                   ``model_cfg.task_type`` / ``model_cfg.deploy.task_type``.
                   Recommended: specify in deploy_config.py as ``task_type = "detection3d"``.
    Returns:
        Pipeline compose object (e.g., mmdet.datasets.transforms.Compose)

    Raises:
    ValueError: If no valid test pipeline found in config or invalid task_type
        ImportError: If required transform packages are not available

    Examples:
        >>> from mmengine.config import Config
        >>> cfg = Config.fromfile('model_config.py')
        >>> pipeline = build_preprocessing_pipeline(cfg, task_type='detection3d')
        >>> # Use pipeline to preprocess data
        >>> results = pipeline({'img_path': 'image.jpg'})
    """
    pipeline_cfg = _extract_pipeline_config(model_cfg)
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be one of {VALID_TASK_TYPES}. "
            f"Please specify a supported task type in the deploy config or function argument."
        )

    logger.info("Building preprocessing pipeline with task_type: %s", task_type)
    try:
        task_cfg = TASK_PIPELINE_CONFIGS[task_type]
    except KeyError:
        raise ValueError(f"Unknown task_type '{task_type}'. " f"Must be one of {VALID_TASK_TYPES}")
    return ComposeBuilder.build(pipeline_cfg=pipeline_cfg, **task_cfg)


def _extract_pipeline_config(model_cfg: Config) -> List[TransformConfig]:
    """
    Extract pipeline configuration from model config.

    Args:
        model_cfg: Model configuration

    Returns:
        List of transform configurations

    Raises:
        ValueError: If no valid pipeline found
    """
    try:
        pipeline_cfg = model_cfg["test_pipeline"]
    except (KeyError, TypeError) as exc:
        raise ValueError("No test pipeline found in config. Expected pipeline at: test_pipeline.") from exc

    if not pipeline_cfg:
        raise ValueError("test_pipeline is defined but empty. Please provide a valid test pipeline.")

    logger.info("Found test pipeline at: test_pipeline")
    return pipeline_cfg
