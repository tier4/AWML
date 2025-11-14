"""
Preprocessing pipeline builder for deployment data loaders.
This module provides functions to extract and build preprocessing pipelines
from MMDet/MMDet3D/MMPretrain configs for use in deployment data loaders.
NOTE: This module is compatible with the new pipeline architecture (BaseDeploymentPipeline).
They serve complementary purposes:
- preprocessing_builder.py: Builds MMDet/MMDet3D preprocessing pipelines for data loaders
- New pipeline architecture: Handles inference pipeline (preprocess → run_model → postprocess)
Data flow: DataLoader (uses preprocessing_builder) → Preprocessed Data → Pipeline (new architecture) → Predictions
See PIPELINE_BUILDER_INTEGRATION_ANALYSIS.md for detailed analysis.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from mmengine.config import Config

logger = logging.getLogger(__name__)

# Valid task types
VALID_TASK_TYPES = ["detection2d", "detection3d", "classification", "segmentation"]


class ComposeBuilder:
    """
    Unified builder for creating Compose objects with different MM frameworks.

    Uses MMEngine-based Compose with init_default_scope for all frameworks.
    """

    @staticmethod
    def build(
        pipeline_cfg: List,
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

        # Import MMEngine components
        try:
            from mmengine.dataset import Compose
            from mmengine.registry import init_default_scope
        except ImportError as e:
            raise ImportError(
                f"Failed to import mmengine components for scope '{scope}'. "
                f"Please ensure mmengine is installed. Error: {e}"
            ) from e

        # Set default scope and build Compose
        try:
            init_default_scope(scope)
            logger.info(f"Building pipeline with mmengine.dataset.Compose (default_scope='{scope}')")
            return Compose(pipeline_cfg)
        except Exception as e:
            raise ImportError(f"Failed to build Compose pipeline for scope '{scope}'. " f"Error: {e}") from e


class PreprocessingPipelineRegistry:
    """
    Registry for preprocessing pipeline builders by task type.

    Provides a clean way to register and retrieve pipeline builders.
    """

    def __init__(self):
        self._builders: Dict[str, Callable[[List], Any]] = {}
        self._register_default_builders()

    def _register_default_builders(self):
        """Register default pipeline builders."""
        self.register("detection2d", self._build_detection2d)
        self.register("detection3d", self._build_detection3d)
        self.register("classification", self._build_classification)
        self.register("segmentation", self._build_segmentation)

    def register(self, task_type: str, builder: Callable[[List], Any]):
        """
        Register a pipeline builder for a task type.
        Args:
            task_type: Task type identifier
            builder: Builder function that takes pipeline_cfg and returns Compose object
        """
        if task_type not in VALID_TASK_TYPES:
            logger.warning(f"Registering non-standard task_type: {task_type}")
        self._builders[task_type] = builder
        logger.debug(f"Registered pipeline builder for task_type: {task_type}")

    def build(self, task_type: str, pipeline_cfg: List) -> Any:
        """
        Build pipeline for given task type.
        Args:
            task_type: Task type identifier
            pipeline_cfg: Pipeline configuration
        Returns:
            Compose object
        Raises:
            ValueError: If task_type is not registered
        """
        if task_type not in self._builders:
            raise ValueError(f"Unknown task_type '{task_type}'. " f"Available types: {list(self._builders.keys())}")
        return self._builders[task_type](pipeline_cfg)

    def _build_detection2d(self, pipeline_cfg: List) -> Any:
        """Build 2D detection preprocessing pipeline."""
        return ComposeBuilder.build(
            pipeline_cfg=pipeline_cfg,
            scope="mmdet",
            import_modules=["mmdet.datasets.transforms"],
        )

    def _build_detection3d(self, pipeline_cfg: List) -> Any:
        """Build 3D detection preprocessing pipeline."""
        return ComposeBuilder.build(
            pipeline_cfg=pipeline_cfg,
            scope="mmdet3d",
            import_modules=["mmdet3d.datasets.transforms"],
        )

    def _build_classification(self, pipeline_cfg: List) -> Any:
        """
        Build classification preprocessing pipeline using mmpretrain.

        Raises:
            ImportError: If mmpretrain is not installed
        """
        return ComposeBuilder.build(
            pipeline_cfg=pipeline_cfg,
            scope="mmpretrain",
            import_modules=["mmpretrain.datasets.transforms"],
        )

    def _build_segmentation(self, pipeline_cfg: List) -> Any:
        """Build segmentation preprocessing pipeline."""
        return ComposeBuilder.build(
            pipeline_cfg=pipeline_cfg,
            scope="mmseg",
            import_modules=["mmseg.datasets.transforms"],
        )


# Global registry instance
_registry = PreprocessingPipelineRegistry()


def build_preprocessing_pipeline(
    model_cfg: Config,
    task_type: Optional[str] = None,
    backend: str = "pytorch",
) -> Any:
    """
    Build preprocessing pipeline from model config.
    This function extracts the test pipeline configuration from a model config
    and builds a Compose pipeline that can be used for preprocessing in deployment data loaders.
    Args:
        model_cfg: Model configuration containing test pipeline definition.
                   Can have pipeline defined in one of these locations:
                   - model_cfg.test_dataloader.dataset.pipeline
                   - model_cfg.test_pipeline
                   - model_cfg.val_dataloader.dataset.pipeline
        task_type: Explicit task type ('detection2d', 'detection3d', 'classification', 'segmentation').
                   Must be provided either via this argument or via
                   ``model_cfg.task_type`` / ``model_cfg.deploy.task_type``.
                   Recommended: specify in deploy_config.py as ``task_type = "detection3d"``.
        backend: Target backend ('pytorch', 'onnx', 'tensorrt').
                 Currently not used, reserved for future backend-specific optimizations.
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
    task_type = _resolve_task_type(model_cfg, task_type)

    logger.info(f"Building preprocessing pipeline with task_type: {task_type}")
    return _registry.build(task_type, pipeline_cfg)


def _resolve_task_type(model_cfg: Config, task_type: Optional[str] = None) -> str:
    """
    Resolve task type from various sources.
    Args:
        model_cfg: Model configuration
        task_type: Explicit task type (highest priority)
    Returns:
        Resolved task type string
    Raises:
        ValueError: If task_type cannot be resolved
    """
    # Priority: function argument > model_cfg.task_type > model_cfg.deploy.task_type
    if task_type is not None:
        _validate_task_type(task_type)
        return task_type

    if "task_type" in model_cfg:
        task_type = model_cfg.task_type
        _validate_task_type(task_type)
        return task_type

    deploy_section = model_cfg.get("deploy", {})
    if isinstance(deploy_section, dict) and "task_type" in deploy_section:
        task_type = deploy_section["task_type"]
        _validate_task_type(task_type)
        return task_type

    raise ValueError(
        "task_type must be specified either via the build_preprocessing_pipeline argument "
        "or by setting 'task_type' in the deploy config (deploy_config.py) or "
        "model config (model_cfg.task_type or model_cfg.deploy.task_type). "
        "Recommended: add 'task_type = \"detection3d\"' (or appropriate type) to deploy_config.py. "
        "Automatic inference has been removed."
    )


def _validate_task_type(task_type: str) -> None:
    """
    Validate task type.
    Args:
        task_type: Task type to validate
    Raises:
        ValueError: If task_type is invalid
    """
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be one of {VALID_TASK_TYPES}. "
            f"Please specify a supported task type in the deploy config or function argument."
        )


def _extract_pipeline_config(model_cfg: Config) -> List:
    """
    Extract pipeline configuration from model config.
    Args:
        model_cfg: Model configuration
    Returns:
        List of transform configurations
    Raises:
        ValueError: If no valid pipeline found
    """
    # Try different possible locations for pipeline config
    pipeline_locations = [
        # Primary location: test_dataloader
        ("test_dataloader", "dataset", "pipeline"),
        # Alternative: direct test_pipeline
        ("test_pipeline",),
        # Fallback: val_dataloader
        ("val_dataloader", "dataset", "pipeline"),
    ]

    for location in pipeline_locations:
        try:
            cfg = model_cfg
            for key in location:
                cfg = cfg[key]
            if cfg:
                logger.info(f"Found test pipeline at: {'.'.join(location)}")
                return cfg
        except (KeyError, TypeError):
            continue

    raise ValueError(
        "No test pipeline found in config. "
        "Expected pipeline at one of: test_dataloader.dataset.pipeline, "
        "test_pipeline, or val_dataloader.dataset.pipeline"
    )


# Public API: Allow custom pipeline builder registration
def register_preprocessing_builder(task_type: str, builder: Callable[[List], Any]):
    """
    Register a custom preprocessing pipeline builder.
    Args:
        task_type: Task type identifier
        builder: Builder function that takes pipeline_cfg and returns Compose object
    Examples:
        >>> def custom_builder(pipeline_cfg):
        ...     # Custom logic
        ...     return Compose(pipeline_cfg)
        >>> register_preprocessing_builder("custom_task", custom_builder)
    """
    _registry.register(task_type, builder)
