import logging
import re
from attr import asdict
from attrs import define, field
from pathlib import Path
from typing import List, Dict, Any, NewType

from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from t4_devkit import Tier4
from t4_devkit.common.box import Box3D as T4Box3D
from typing_extensions import Self

from tools.detection3d.create_data_t4dataset import get_scene_root_dir_path


@define
class BBox3D:
    """3D bounding box representation.

    Attributes:
        translation (Dict[str, float]): Dictionary containing x, y, z coordinates of the box center
        velocity (Dict[str, float]): Dictionary containing x, y, z components of velocity
        acceleration (Dict[str, float]): Dictionary containing x, y, z components of acceleration
        size (Dict[str, float]): Dictionary containing width, length, height of the box
        rotation (Dict[str, float]): Dictionary containing quaternion components w, x, y, z for rotation
    """
    translation: Dict[str, float] = field()
    velocity: Dict[str, float] = field()
    acceleration: Dict[str, float] = field()
    size: Dict[str, float] = field()
    rotation: Dict[str, float] = field()

    @classmethod
    def from_t4box3d(cls, t4box3d: T4Box3D) -> Self:
        """Create a BBox3D instance from a T4Box3D object.

        Args:
            t4box3d (T4Box3D): Input box object containing center, velocity, dimensions, and orientation

        Returns:
            Self: New instance of BBox3D with converted data

        Note:
            Acceleration is initialized as zero for all components
        """
        translation: Dict[str, float] = dict(zip(["x", "y", "z"], t4box3d.center))
        velocity: Dict[str, float] = dict(zip(["x", "y", "z"], t4box3d.velocity))
        acceleration: Dict[str, float] = dict(zip(["x", "y", "z"], [0.0, 0.0, 0.0]))
        size: Dict[str, float] = dict(zip(["width", "length", "height"], t4box3d.wlh))
        rotation: Dict[str, float] = dict(zip(["w", "x", "y", "z"], t4box3d.orientation.q.tolist()))
        
        return cls(
            translation=translation,
            velocity=velocity,
            acceleration=acceleration,
            size=size,
            rotation=rotation,
        )

@define
class ObjectAnnotation:
    """Base class for object annotations.

    Attributes:
        category_name (str): Name of the object category
        instance_id (str): Unique identifier for the instance
        attribute_names (List[str]): List of attribute names associated with the object
        num_lidar_pts (int): Number of LiDAR points on the object
        num_radar_pts (int): Number of radar points on the object
    """
    category_name: str = field()
    instance_id: str = field()
    attribute_names: List[str] = field()
    num_lidar_pts: int = field()
    num_radar_pts: int = field()

@define
class ObjectAnnotation3D(ObjectAnnotation):
    """3D object annotation, extending base ObjectAnnotation.

    Attributes:
        three_d_bbox (BBox3D): 3D bounding box information for the object
    """
    three_d_bbox: BBox3D = field()

@define
class FrameAnnotations:
    """Collection of object annotations for a single frame.

    Attributes:
        objects (List[ObjectAnnotation3D]): List of dictionary objects containing annotation data. index: instance_id
    """
    objects: List[ObjectAnnotation3D] = field(factory=list)

@define
class SceneAnnotations:
    """Collection of object annotations for a single scene.

    Reference:
        scene_anno_dict in AnnotationFilesGenerator. This is the interface between PseudoLabelGenerator3D and AnnotationFilesGenerator.
        https://github.com/tier4/tier4_perception_dataset/blob/2698f8fdf38dd10e4402c049e5330b40fe5d55fd/perception_dataset/t4_dataset/annotation_files_generator.py#L178

    Attributes:
        scene_id (str): Name of the scene
        scene_annotations_dict (Dict[int, SceneAnnotations]): Dictionary mapping frame ID to their annotations. key: frame_id
    """
    scene_id: str = field()
    scene_annotations_dict: Dict[int, FrameAnnotations] = field(factory=dict)

    def to_scene_anno_dict(self) -> Dict[int, List[Dict[str, Any]]]:
        """Convert to scene_anno_dict"""
        raw_dict = asdict(self, dict_factory=dict)
        return {
            frame_id: scene_annotations["objects"]
            for frame_id, scene_annotations in raw_dict["scene_annotations_dict"].items()
        }

def _is_non_annotated(data_root: Path) -> bool:
   """Check if t4dataset in data_root is non-annotated t4dataset.
   
   Args:
       data_root (Path): Path to the dataset directory containing the generated annotation files.
           e.g, "./data/t4dataset/pseudo_xx1/scene_0/"

   Returns:
       bool: True if directory contains empty annotation files, False otherwise

   Raises:
       FileNotFoundError: If annotation directory or required files do not exist
       
   Note:
       - Check if sample_annotation.json, instance.json and attribute.json are empty in data_root / "annotation".
   """
   annotation_dir = data_root / "annotation"
   if not annotation_dir.exists():
       raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

   # Required annotation files
   json_files = [
       "sample_annotation.json",
       "instance.json", 
       "attribute.json"
   ]

   # Check all json files exist and are empty
   for filename in json_files:
       file_path = annotation_dir / filename
       
       # Check if file exists
       if not file_path.exists():
           raise FileNotFoundError(f"Required annotation file not found: {file_path}")
           
       # Check if file has empty JSON structure
       with open(file_path) as f:
           content = f.read().strip()
           if content not in ['{}', '[]']:
               return False

   return True

class PseudoLabelGenerator3D:
    def __init__(
        self,
        non_annotated_dataset_path: Path,
        scene_ids: List[str],
        t4dataset_config: Dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        """
        Wrapper class for generating pseudo label using tier4_perception_dataset.

        Args:
            non_annotated_dataset_path (Path): Path to the root directory of dataset. e.g, "./data/t4dataset/pseudo_xx1/"
            scene_ids: (List[str]): Names of scene in non_annotated_dataset_path / version
            t4dataset_config: (Dict[str, Any]): Config for generating T4dataset.
            logger (logging.Logger): Logger instance for output messages.

        Attributes:
            dataset_annotations (Dict[str, SceneAnnotations]): Dictionary storing annotations for each dataset

        Raises:
            FileNotFoundError: If the specified non_annotated_dataset_path directory does not exist
        """
        self.non_annotated_dataset_path = Path(non_annotated_dataset_path)
        self.t4dataset_config: Dict[str, Any] = t4dataset_config
        self.logger: logging.Logger = logger

        if not self.non_annotated_dataset_path.exists():
            raise FileNotFoundError(f"Database directory is not found: {self.non_annotated_dataset_path}")

        self.dataset_annotations: Dict[str, SceneAnnotations] = {
            scene_id: SceneAnnotations(scene_id=scene_id, scene_annotations_dict={})
            for scene_id in scene_ids
        }

    def add_3d_annotation_object(self, scene_id: str, frame_id: int, instance_id: str, category_name: str, t4box3d: T4Box3D) -> None:
        """Adds a 3D object annotation to the dataset.

        Args:
            scene_id (str): Name of the scene. e.g, "scene_0"
            frame_id (int): Frame id. e.g, 0
            instance_id (str): Instance ID of the object. e.g, "fade3eb7-77b4-420f-8248-b532800388a3"
            category_name (str): Category name of the object. e.g, "truck"
            t4box3d (T4Box3D): 3D bounding box information
        """

        object_annotation = ObjectAnnotation3D(
            category_name=category_name,
            instance_id=instance_id,
            attribute_names=[],
            three_d_bbox=BBox3D.from_t4box3d(t4box3d),
            num_lidar_pts=0, # num_lidar_pts will be filled later by AnnotationFilesGenerator.
            num_radar_pts=0,
        )

        if frame_id not in self.dataset_annotations[scene_id].scene_annotations_dict:
            self.dataset_annotations[scene_id].scene_annotations_dict[frame_id] = FrameAnnotations(objects=[object_annotation])
        else:
            self.dataset_annotations[scene_id].scene_annotations_dict[frame_id].objects.append(object_annotation)

    def dump(self, overwrite: bool) -> None:
        """Save the generated annotations as a T4 dataset format.

        Args:
            overwrite (bool): If True, this code can overwrite sample_annotation.json even if t4dataset in non_annotated_dataset_path already have the annotation information.

        Note:
            - Converts annotations to T4 dataset format for each scene using AnnotationFilesGenerator
            - Validates the generated files after conversion
        """
        self.logger.info("Saving t4dataset...")
        for scene_id, dataset_annotation in self.dataset_annotations.items():

            input_dir = Path(get_scene_root_dir_path(self.non_annotated_dataset_path.parent, self.non_annotated_dataset_path.name, scene_id,))

            if not overwrite:
                # check if t4dataset in input_dir is non-annotated t4dataset
                if not _is_non_annotated(data_root=input_dir):
                    raise RuntimeError(
                        f"T4dataset in {input_dir} already contains annotations. "
                        "Use --overwrite flag to overwrite existing annotations."
                    )

            # dump t4dataset
            scene_anno_dict: Dict[int, List[Dict[str, Any]]] = dataset_annotation.to_scene_anno_dict()
            annotation_files_generator = AnnotationFilesGenerator(description=self.t4dataset_config)
            annotation_files_generator.convert_one_scene(
                input_dir=input_dir,
                output_dir=input_dir,
                scene_anno_dict=scene_anno_dict,
                dataset_name=scene_id,
            )
            self.logger.info("Pseudo labeled t4dataset is saved at {}".format(input_dir))
            
            # validate t4dataset
            self._validate(data_root=input_dir)

        self.logger.info("Finish generating t4dataset.")

    def _validate(self, data_root: Path) -> None:
        """Validate the generated T4 dataset using t4-devkit.

        Args:
            data_root (Path): Path to the dataset directory containing the generated annotation files. e.g, "./data/t4dataset/pseudo_xx1/scene_0/"

        Note:
            - Uses Tier4 class from t4-devkit for validation
            - Does not raise exceptions but logs errors for failed validations
        """
        self.logger.info("Validate pseudo labeled t4dataset.")
        try:
            Tier4(
                version="annotation",
                data_root=data_root,
            )
            self.logger.info(f"Validation for pseudo labeled dataset in {data_root} succeeded.")
        except Exception as e:
            self.logger.error(
                f"Validation for pseudo labeled dataset failed. Please check generated dataset in {data_root} by t4-devkit. Error message: {str(e)}"
            )
