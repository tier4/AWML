"""Shared backend-executor primitives for point-cloud 3D detectors.

Point-cloud detectors (CenterPoint, BEVFusion, …) all feed the network the same per-sample
input: the raw ``points`` tensor plus the ``metainfo`` needed by postprocess. This base
implements that single shared ``prepare_input`` so each project executor only has to provide
the backend-specific ``create_pipeline`` (and optionally ``get_output_names``).
"""

from __future__ import annotations

from typing import Any, Mapping

from typing_extensions import override

from deployment.execution.backend_executor import BackendExecutor
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import InferenceInput


class PointCloudBackendExecutor(BackendExecutor):
    """BackendExecutor whose model input is ``(points, metainfo)`` (shared prepare_input)."""

    @override
    def prepare_input(
        self,
        sample: Mapping[str, Any],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build InferenceInput from a sample's ``points`` + ``metainfo``.

        Args:
            sample: Dict with 'points' and 'metainfo'.
            data_loader: Unused; kept for interface compatibility.
            device: Unused; kept for interface compatibility.

        Raises:
            ValueError: If 'points' is missing from sample.
            KeyError: If 'metainfo' is missing from sample.
        """
        if "points" not in sample:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        if "metainfo" not in sample:
            raise KeyError("Sample must contain 'metainfo' for postprocess.")
        return InferenceInput(data=sample["points"], metadata=sample["metainfo"])
