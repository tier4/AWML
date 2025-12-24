"""
Project Names for Pipeline Registry.

Usage:
    from deployment.pipelines.common.project_names import ProjectNames

    # In factory:
    class CenterPointPipelineFactory(BasePipelineFactory):
        @classmethod
        def get_project_name(cls) -> str:
            return ProjectNames.CENTERPOINT

    # When creating pipeline:
    PipelineFactory.create(ProjectNames.CENTERPOINT, model_spec, pytorch_model)
"""


class ProjectNames:
    """
    Constants for project names.

    Add new project names here when adding new projects.
    """

    CENTERPOINT = "centerpoint"
    YOLOX = "yolox"
    CALIBRATION = "calibration"

    @classmethod
    def all(cls) -> list:
        """Return all defined project names."""
        return [
            value
            for key, value in vars(cls).items()
            if not key.startswith("_") and isinstance(value, str) and key.isupper()
        ]
