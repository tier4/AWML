from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from attrs import define, field
from filterpy.kalman import KalmanFilter
from numpy.typing import NDArray
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from t4_devkit.dataclass import Box3D as T4Box3D
from t4_devkit.dataclass import SemanticLabel, Shape, ShapeType

# Define class matching rules
CLASS_MATCHING_MATRIX = {
    "car": ["car", "truck", "bus"],
    "truck": ["car", "truck", "bus"],
    "bus": ["car", "truck", "bus"],
    "bicycle": ["bicycle", "pedestrian"],
    "pedestrian": ["bicycle", "pedestrian"],
}

# Class-specific tracking parameters
#
# Parameters for each class (car/truck/bus/bicycle/pedestrian):
#   measurement_noise_scale: Scale for Kalman filter measurement noise (R)
#     - Higher values for larger objects (truck/bus: 8.0, car: 5.0, bicycle: 5.0, pedestrian: 3.0)
#     - Represents uncertainty in detection measurements
#
#   process_noise_scale: Scale for Kalman filter process noise (Q)
#     - Higher for vehicles (car/truck/bus: 0.3-0.35) to handle dynamic motion
#     - Lower for bicycle/pedestrian (0.05/0.03) for smoother tracking
#
#   initial_uncertainty_scale: Initial state uncertainty scale (P)
#     - Matched with measurement noise scale for each class
#     - Higher values allow more rapid convergence to measurements
#
#   initial_velocity_uncertainty_scale: Initial velocity uncertainty scale
#     - Higher for bicycle/pedestrian (500.0/300.0) due to irregular motion
#     - Lower for vehicles (50.0) with more predictable motion patterns
#
#   max_age: Maximum frames to keep track without updates
#     - Set to 100 for all classes for long-term tracking stability
#
#   distance_threshold: Maximum distance for valid detection-track matching (meters)
#     - Larger for vehicles (car: 12.0, truck/bus: 15.0) due to size
#     - Smaller for bicycle/pedestrian (3.0) for more precise matching
#
CLASS_PARAMS = {
    "car": {
        "measurement_noise_scale": 5.0,
        "process_noise_scale": 0.3,
        "initial_uncertainty_scale": 5.0,
        "initial_velocity_uncertainty_scale": 50.0,
        "max_age": 100,
        "distance_threshold": 12.0,
    },
    "truck": {
        "measurement_noise_scale": 8.0,
        "process_noise_scale": 0.35,
        "initial_uncertainty_scale": 8.0,
        "initial_velocity_uncertainty_scale": 50.0,
        "max_age": 100,
        "distance_threshold": 15.0,
    },
    "bus": {
        "measurement_noise_scale": 8.0,
        "process_noise_scale": 0.35,
        "initial_uncertainty_scale": 8.0,
        "initial_velocity_uncertainty_scale": 50.0,
        "max_age": 100,
        "distance_threshold": 15.0,
    },
    "bicycle": {
        "measurement_noise_scale": 5.0,
        "process_noise_scale": 0.05,
        "initial_uncertainty_scale": 5.0,
        "initial_velocity_uncertainty_scale": 500.0,
        "max_age": 100,
        "distance_threshold": 3.0,
    },
    "pedestrian": {
        "measurement_noise_scale": 3.0,
        "process_noise_scale": 0.03,
        "initial_uncertainty_scale": 3.0,
        "initial_velocity_uncertainty_scale": 300.0,
        "max_age": 100,
        "distance_threshold": 3.0,
    },
}

# in cost matrix, use this number instead of inf.
LARGE_VALUE: float = 1e9


def _transform_pred_instance_to_global_t4box(
    bbox3d: List[float],
    velocity: List[float],
    confidence: float,
    label: str,
    instance_id: str,
    ego2global: NDArray,
    timestamp: float,
) -> T4Box3D:
    """Convert a detection instance to T4Box3D format and transform to global coordinates.

    Args:
        bbox3d (List[float]): 3D bounding box parameters [x, y, z, l, w, h, yaw]
            - x, y, z: Center position in ego vehicle coordinates
            - l (length): Size along x-axis in object coordinates
            - w (width): Size along y-axis in object coordinates
            - h (height): Size along z-axis in object coordinates
            - yaw: Rotation angle around z-axis in radians
        velocity (List[float]): Object velocity vector [vx, vy] in ego coordinates
        confidence (float): Detection confidence confidence [0-1]
        label (str): Object class label. e.g, "bus"
        instance_id (str): Instance ID of the object. e.g, "fade3eb7-77b4-420f-8248-b532800388a3"
        ego2global (NDArray): 4x4 transformation matrix from ego to global coordinates
            - 3x3 rotation matrix in top-left
            - Translation vector in fourth column
            - Last row is [0, 0, 0, 1]
        timestamp (float): unix timestamp. e.g, 1711672980.049259

    Returns:
        T4Box3D: 3D bounding box in T4Box3D format, transformed to global coordinates

    Example:
        >>> # bbox parameters in ego coordinates [x, y, z, l, w, h, yaw]
        >>> bbox3d = [-7.587669372558594, 5.918113708496094, 0.3056640625, 11.3125, 2.66796875, 3.30078125, -0.07647705078125]
        >>>
        >>> # velocity vector [vx, vy]
        >>> velocity = [3.708984375, -0.304443359375]
        >>>
        >>> # detection confidence and class label
        >>> confidence = 0.5257580280303955
        >>> label = "bus"
        >>>
        >>> # detection confidence and class label
        >>> instance_id = "fade3eb7-77b4-420f-8248-b532800388a3"
        >>> timestamp = 1711672980.049259
        >>>
        >>> # transformation matrix from ego to global coordinates
        >>> ego2global = np.array([
        ...     [-7.52366364e-01,  6.58483088e-01, -1.85688175e-02,  2.67971660e+04],
        ...     [-6.58744812e-01, -7.52079487e-01,  2.07780898e-02,  2.95396055e+04],
        ...     [-2.83205271e-04,  2.78648492e-02,  9.99611676e-01,  4.27038288e+00],
        ...     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ... ])
        >>>
        >>> # convert to T4Box3D and transform to global coordinates
        >>> box = _transform_pred_instance_to_global_t4box(bbox3d, velocity, confidence, label, instance_id, ego2global, timestamp)

    Note:
        - Input box must be in ego vehicle coordinates
        - Object dimensions are defined in object coordinates:
            * Length (l): Size along object's x-axis
            * Width (w): Size along object's y-axis
            * Height (h): Size along object's z-axis
        - Vertical velocity (vz) is automatically set to 0.0
        - T4Box3D is transformed to global coordinates using ego2global matrix
    """
    # [x, y, z]
    position: List[float] = bbox3d[:3]
    # quaternion
    rotation = Quaternion(axis=[0, 0, 1], radians=bbox3d[6])
    # [w, l, h]
    shape = Shape(shape_type=ShapeType.BOUNDING_BOX, size=(bbox3d[4], bbox3d[3], bbox3d[5]))
    # [vx, vy, vz]
    velocity: Tuple[float] = (*velocity, np.float64(0.0))

    box = T4Box3D(
        unix_time=int(timestamp),
        frame_id="base_link",
        semantic_label=SemanticLabel(label),
        position=position,
        rotation=rotation,
        shape=shape,
        velocity=velocity,
        confidence=confidence,
        uuid=instance_id,
    )

    # Transform box to global coord system
    box.rotate(Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
    box.translate(ego2global[:3, 3])
    box.frame_id = "map"

    return box


@define(frozen=True)
class TrackedBox2D:
    """Tracked bbox class

    Attributes:
        position_xy (NDArray): Center position in map coordinates.
        class_name (str): Name of the object category
        size_wl (NDArray): Size in object coordinates
            - w (width): Size along y-axis
            - l (length): Size along x-axis

    Note:
        position_xy is set to be used for calculating BEV center distance.
        size_wl is set to be used for calculation BEV IoU.
        size_wl is optional, so please set when you want to use BEV IoU as matching metrics.
    """

    position_xy: NDArray = field()
    class_name: str = field()
    size_wl: NDArray | None = field(default=None, init=False)


class KalmanBoxTracker:
    """2D Kalman filter for tracking objects in Bird's Eye View.

    This tracker only tracks x, y positions and velocities in the BEV plane,
    ignoring height information. It uses a constant velocity motion model
    with adjustable parameters per object class.

    Attributes:
        count (int): Global counter for tracking IDs
        prev_timestamp (float): Previous timestamp for time delta calculation
        latest_timestamp (float): Latest timestamp for prediction
        id (str): Unique instance ID for this tracker
        class_name (str): Object class name (e.g. "car", "pedestrian")
        age (int): Number of frames since last update
        max_age (int): Maximum allowed frames without update
        kf (KalmanFilter): Kalman filter for state estimation
            State vector [x, y, vx, vy]:
                - x, y: Object center position in map coordinate
                - vx, vy: Object velocity in map coordinate
    """

    count: int = 0

    def __init__(self, t4box3d: T4Box3D, class_params=CLASS_PARAMS):
        """
        Initialize Kalman filter with class-specific parameters.

        Args:
            t4box3d (T4Box3D): 3D bounding box
            class_params: Dictionary of class-specific tracking parameters:
                - measurement_noise_scale: Scale for measurement uncertainty
                - process_noise_scale: Scale for motion model uncertainty
                - initial_uncertainty_scale: Scale for initial state uncertainty
                - initial_velocity_uncertainty_scale: Scale for initial velocity uncertainty
                - max_age: Maximum frames without update before deletion
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=4)

        self.prev_timestamp: float = t4box3d.unix_time
        self.latest_timestamp: float = t4box3d.unix_time
        self.id: str = t4box3d.uuid

        # Measurement matrix (identity matrix for direct state observation). States are x,y,vx,vy
        self.kf.H = np.eye(4)

        # Set class-specific parameters
        self.class_name: str = t4box3d.semantic_label.name
        params = class_params[self.class_name]
        self.kf.R *= params["measurement_noise_scale"]
        self.kf.Q *= params["process_noise_scale"]
        self.kf.P *= params["initial_uncertainty_scale"]
        self.kf.P[2:, 2:] *= params["initial_velocity_uncertainty_scale"]

        # Set life cycle management param
        self.age: int = 0
        self.max_age: int = params["max_age"]

        # Initialize state
        self.kf.x = np.concatenate([t4box3d.position[:2], t4box3d.velocity[:2]]).reshape((4, 1))

        KalmanBoxTracker.count += 1

    def predict(self, timestamp: float) -> TrackedBox2D:
        """
        Predict next state using constant velocity model.

        Args:
            timestamp (float): Current timestamp

        Returns:
            TrackedBox2D: Predicted 2D box containing predicted position and class name.
        """
        time_lag = timestamp - self.prev_timestamp
        # State transition matrix (uniform linear motion model)
        self.kf.F = np.array(
            [
                [1, 0, time_lag, 0],  # x = x + vx * time_lag
                [0, 1, 0, time_lag],  # y = y + vy * time_lag
                [0, 0, 1, 0],  # vx
                [0, 0, 0, 1],  # vy
            ]
        )

        self.kf.predict()
        self.age += 1
        self.latest_timestamp = timestamp

        tracked_box = TrackedBox2D(position_xy=self.kf.x[:2].reshape((2,)), class_name=self.class_name)
        return tracked_box

    def update(self, t4box3d: T4Box3D):
        """
        Update state with new measurement.

        Updates Kalman filter state with new position and velocity observations.
        Resets age counter and updates timestamp.

        Args:
            t4box3d (T4Box3D): Observed 3D bounding box
        """
        self.age = 0

        # Update Kalman filter with new measurement
        self.kf.update(np.concatenate([t4box3d.position[:2], t4box3d.velocity[:2]]).reshape((4, 1)))

        self.prev_timestamp = t4box3d.unix_time

    def end_of_life(self):
        """Check if tracker should be terminated.

        Returns:
            True if tracker has not been updated for max_age frames
        """
        return self.age >= self.max_age


def associate_detections_to_tracks(
    bbox_list: List[T4Box3D],
    tracked_box_list: List[TrackedBox2D],
    class_params: Dict[str, Dict[str, int | float]] = CLASS_PARAMS,
    class_matching_matrix: Dict[str, List[str]] = CLASS_MATCHING_MATRIX,
    large_value: float = LARGE_VALUE,
) -> Tuple[NDArray, set]:
    """Match detections with tracked boxes.

    Args:
        bbox_list (List[T4Box3D]): Detected 3D boxes
        tracked_box_list (List[TrackedBox2D]): Tracked 2D boxes

    Returns:
        - NDArray: Valid matches as [[det_idx, trk_idx], ...]
        - set: Unmatched detection indices
    """

    def _calculate_cost_matrix(
        bbox_list: List[T4Box3D],
        tracked_box_list: List[TrackedBox2D],
        large_value=large_value,
        class_params=class_params,
    ) -> Tuple[NDArray, bool]:
        """Calculate cost matrix for detection-to-track association.

        Builds a cost matrix for bipartite matching between detections and tracks.
        Cost is based on euclidean distance between centers in BEV (bird's eye view).
        Only calculates costs between class-compatible pairs (e.g. car-car, pedestrian-bicycle).

        Args:
            bbox_list: List of detected 3D boxes
            tracked_box_list: List of predicted 2D tracked boxes
            large_value: Value to use for invalid matches (default: 1e9)
            class_params: Dictionary of class parameters containing distance thresholds
                Format: {"class_name": {"distance_threshold": float, ...}}

        Returns:
            Tuple containing:
                - cost_matrix (NDArray): Matrix of shape (num_detections, num_tracks)
                    containing matching costs or large_value for invalid matches
                - has_valid_pairs (bool): Whether any valid detection-track pairs exist

        Note:
            A match is considered valid if:
            1. The classes are compatible (defined in CLASS_MATCHING_MATRIX)
            2. The center distance is below the class-specific threshold
        """
        cost_matrix = np.full((len(bbox_list), len(tracked_box_list)), large_value)
        has_valid_pairs = False

        for i, det_box in enumerate(bbox_list):
            det_class = det_box.semantic_label.name
            det_position = det_box.position[:2]
            cost_threshold = class_params[det_class]["distance_threshold"]

            for j, trk_box in enumerate(tracked_box_list):
                # Skip if classes cannot match
                if trk_box.class_name not in class_matching_matrix[det_class]:
                    continue

                # Calculate cost only if classes match
                cost = np.linalg.norm(det_position - trk_box.position_xy)

                # Update cost matrix if within threshold
                if cost <= cost_threshold:
                    cost_matrix[i, j] = cost
                    has_valid_pairs = True

        return cost_matrix, has_valid_pairs

    # Empty case handling
    if not tracked_box_list or not bbox_list:
        return np.array([]), set(range(len(bbox_list)))

    # Calculate cost matrix and check valid pairs
    cost_matrix, has_valid_pairs = _calculate_cost_matrix(bbox_list, tracked_box_list)
    if not has_valid_pairs:
        return np.array([]), set(range(len(bbox_list)))

    # Assignment
    det_indices, trk_indices = linear_sum_assignment(cost_matrix)

    # Filter valid matches
    valid_matches = np.array(
        [
            [det_idx, trk_idx]
            for det_idx, trk_idx in zip(det_indices, trk_indices)
            if cost_matrix[det_idx, trk_idx] < large_value
        ]
    )

    # Calculate unmatched detections
    matched_dets = set(match[0] for match in valid_matches)
    unmatched_dets = set(range(len(bbox_list))) - matched_dets

    return valid_matches, unmatched_dets


class MOTModel:
    """Multi-Object Tracker using Kalman filters in 2D BEV space.

    Manages multiple KalmanBoxTracker instances to track objects across frames.
    Performs data association between detections and existing tracks using
    bipartite matching based on spatial distance and class compatibility.

    Attributes:
        classes (List[str]): List of trackable object classes
        trackers (List[KalmanBoxTracker]): List of active trackers
    """

    def __init__(self, classes: List[str]):
        """
        Initialize multi-object tracker.

        Args:
            classes: List of class names to track (e.g. ["car", "pedestrian"])
        """
        self.classes = classes
        self.trackers = []

    def _convert_pred_instance_to_global_t4boxes(
        self, pred_instances_3d, ego2global, timestamp
    ) -> Tuple[List[str], List[T4Box3D]]:
        """Convert detection results to T4Box3D format in global coordinates.

        Args:
            pred_instances_3d: List of detection instances, each containing:
                - bbox_3d (List[float]): 3D bounding box parameters [x,y,z,l,w,h,yaw]
                - velocity (List[float]): Object velocity [vx,vy]
                - bbox_score_3d (float): Detection confidence score
                - bbox_label_3d (int): Class labels, e.g, 0
                - instance_id_3d (str): Instance ID
            ego2global: 4x4 transformation matrix from ego to global coordinates
            timestamp: Current frame timestamp

        Returns:
            Tuple containing:
                - instance_ids (List[str]): List of instance IDs from detections
                - bbox_list (List[T4Box3D]): List of 3D boxes transformed to global coordinates
        """
        instance_ids: List[str] = []
        bbox_list: List[T4Box3D] = []
        for instance in pred_instances_3d:
            instance_ids.append(instance["instance_id_3d"])
            bbox_list.append(
                _transform_pred_instance_to_global_t4box(
                    instance["bbox_3d"],
                    instance["velocity"],
                    instance["bbox_score_3d"],
                    self.classes[instance["bbox_label_3d"]],
                    instance["instance_id_3d"],
                    ego2global,
                    timestamp,
                ),
            )

        return instance_ids, bbox_list

    def _add_new_tracker(self, bbox_list: List[T4Box3D], det_ids: Optional[Iterable[int]] = None):
        """
        Add new trackers for specified detections.

        Args:
            bbox_list (List[T4Box3D]): Detected 3D boxes
            det_ids : Indices of detections to create trackers for.
                If None, creates trackers for all detections.
        """
        if det_ids is None:
            det_ids = range(len(bbox_list))
        for i in det_ids:
            self.trackers.append(KalmanBoxTracker(bbox_list[i]))

    def _remove_old_tracker(self):
        """Remove trackers that have exceeded their maximum age.

        A tracker is considered old if it hasn't been updated for max_age frames.
        """
        self.trackers = [trk for trk in self.trackers if not trk.end_of_life()]

    def frame_mot(self, pred_instances_3d: List[Dict[str, Any]], ego2global: NDArray, timestamp: float) -> List[str]:
        """
        Update tracking for the current frame.

        Args:
            pred_instances_3d: Dictionary containing detection results
                - bbox_3d (List[float]): 3D bounding boxes [x,y,z,l,w,h,yaw]
                - velocity (List[float]): Object velocities [vx,vy]
                - bbox_score_3d (float): Confidence scores
                - bbox_label_3d (int): Class labels, e.g, 0
                - instance_id_3d (str): Instance IDs
            ego2global: (4, 4) Transformation matrix from ego to global coordinates
            timestamp: Current frame timestamp

        Returns:
            List[str]: Updated instance IDs for all detections
        """
        # Convert pred_instance to bbox in map coordinates
        instance_ids, bbox_list = self._convert_pred_instance_to_global_t4boxes(
            pred_instances_3d, ego2global, timestamp
        )

        # Initialize new trackers for all detections if no existing trackers
        if not len(self.trackers):
            self._add_new_tracker(bbox_list)
            return instance_ids

        # Get tracked_box_list from active trackers
        tracked_box_list: List[TrackedBox2D] = [trk.predict(timestamp) for trk in self.trackers]

        # Matching
        valid_matches, unmatched_dets = associate_detections_to_tracks(bbox_list, tracked_box_list)

        # For unmatched detections, add new trackes
        self._add_new_tracker(bbox_list, unmatched_dets)

        # Update matched trackers and instance IDs
        for det_idx, trk_idx in valid_matches:
            # Update matched tracker
            trk = self.trackers[trk_idx]
            trk.update(bbox_list[det_idx])

            # Update instance id
            instance_ids[det_idx] = trk.id

        # Remove the trackers which come to the end of life.
        self._remove_old_tracker()

        return instance_ids
