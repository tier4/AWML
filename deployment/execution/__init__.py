"""Backend execution primitives. Import concrete submodules (``deployment.execution.backend_executor``, …).

Execution is the shared stage that both evaluation and verification build on: a ``BackendExecutor``
turns a sample into an ``InferenceInput`` and runs it through a backend pipeline. It has no
dependency on the metrics (evaluation) or comparison (verification) layers.
"""
