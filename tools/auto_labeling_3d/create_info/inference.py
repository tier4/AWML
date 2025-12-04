import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.config import Config
from mmengine.device import get_device
from mmengine.runner import Runner, autocast, load_checkpoint
from mmengine.utils import ProgressBar
from numpy.typing import NDArray


def get_identifier_from_sample(sample: Dict[str, Any]) -> str:
    """
    Create unique identifier from sample data.
    
    Args:
        sample (Dict[str, Any]): Sample data containing token, sample_idx, and/or lidar_path
        
    Returns:
        str: Unique identifier for the sample
    """
    sample_token = sample.get('token', None)
    sample_idx = sample.get('sample_idx', None)
    lidar_path = sample.get('lidar_path', None)
    
    # Use token as primary identifier, fallback to sample_idx_lidar_path
    identifier = sample_token or f"{sample_idx}_{lidar_path}"
    return identifier


def _predict_one_frame(model: Any, data: Dict[str, Any]) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Predict pseudo label for one frame.
    """
    with autocast(enabled=True):
        outputs: List = model.test_step(data)

    # get bboxes, scores, labels
    if hasattr(outputs[0], "pred_instances_3d"):
        bboxes: NDArray = outputs[0].pred_instances_3d["bboxes_3d"].tensor.detach().cpu()
        scores: NDArray = outputs[0].pred_instances_3d["scores_3d"].detach().cpu().numpy()
        labels: NDArray = outputs[0].pred_instances_3d["labels_3d"].detach().cpu().numpy()
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        box_dims: NDArray = bboxes.dims.numpy().astype(np.float64)
        box_gravity_centers: NDArray = bboxes.gravity_center.numpy().astype(np.float64)

    else:
        bboxes: NDArray = outputs[0]["pred_instances_3d"]["bboxes_3d"].tensor.detach().cpu()
        scores: NDArray = outputs[0]["pred_instances_3d"]["scores_3d"].detach().cpu().numpy()
        labels: NDArray = outputs[0]["pred_instances_3d"]["labels_3d"].detach().cpu().numpy()
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        box_dims: NDArray = bboxes.dims.numpy().astype(np.float64)
        box_gravity_centers: NDArray = bboxes.center.numpy().astype(np.float64)

    box_yaws: NDArray = bboxes.yaw.numpy().astype(np.float64)
    box_velocities: NDArray = bboxes.tensor.numpy().astype(np.float64)[:, 7:9]

    bboxes = np.concatenate(
        [
            # (N, 3) [x, y, z]
            box_gravity_centers,
            # (N, 3) [x_size(length), y_size(width), z_size(height)]
            box_dims,
            # (N, 1) [yaw]
            box_yaws[:, None],
            # (N, 2) [vx, vy]
            box_velocities,
        ],
        axis=1,
    )

    return bboxes, scores, labels


def get_filtered_data_mapping(dataset_instance, dataloader) -> Dict[str, int]:
    """
    Get mapping between original sample identifiers and filtered indices.
    
    Returns:
        Dict[str, int]: Maps original sample identifier to filtered index
    """
    print(f"Dataset type: {type(dataset_instance)}")
    print(f"Dataset length: {len(dataset_instance)}")
    print(f"Extracting sample mapping from dataloader...")
    original_to_filtered_map = {}
    
    # Create a temporary iterator to extract sample metadata
    temp_dataloader = iter(dataloader)
    
    for filtered_idx in range(len(dataloader)):
        batch_data = next(temp_dataloader)
        
        # Get data info for this filtered index
        data_info = dataset_instance.get_data_info(filtered_idx)
        
        # Create unique identifier for matching
        identifier = get_identifier_from_sample(data_info)
        original_to_filtered_map[identifier] = filtered_idx
    # Verify lengths
    print(f"Dataloader length: {len(dataloader)}")
    print(f"Mapping created for: {len(original_to_filtered_map)} samples")
    
    if len(original_to_filtered_map) != len(dataloader):
        raise ValueError(f"Mapping mismatch - dataloader: {len(dataloader)}, mapping: {len(original_to_filtered_map)}")

    return original_to_filtered_map


def _results_to_info(
    non_annotated_dataset_info: Dict[str, Any],
    inference_results: List[Tuple[NDArray, NDArray, NDArray]],
    original_to_filtered_map: Dict[str, int],
) -> Dict[str, Any]:
    """
    Convert non annotated dataset info and inference results to pseudo labeled info.
    """
    # Create pseudo labeled dataset info using original metadata
    pseudo_labeled_dataset_info = {
        "metainfo": non_annotated_dataset_info.get("metainfo", {}),
        "data_list": []
    }
    
    # Get original data_list
    original_data_list = non_annotated_dataset_info["data_list"]
    
    # Process all original samples
    for frame_index, original_sample in enumerate(original_data_list):
        
        # Create identifier for this original sample
        identifier = get_identifier_from_sample(original_sample)
        
        if identifier in original_to_filtered_map:
            # This sample has inference results
            filtered_idx = original_to_filtered_map[identifier]
            
            bboxes, scores, labels = inference_results[filtered_idx]
            
            pred_instances_3d: List[Dict[str, Any]] = []
            for instance_index, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
                pred_instance_3d: Dict[str, Any] = {}

                # [x, y, z, x_size(length), y_size(width), z_size(height), yaw]
                pred_instance_3d["bbox_3d"]: List[float] = bbox[:7].tolist()
                # [vx, vy]
                pred_instance_3d["velocity"]: List[float] = bbox[7:9].tolist()
                pred_instance_3d["instance_id_3d"]: str = str(uuid.uuid4())
                pred_instance_3d["bbox_label_3d"]: int = int(label)
                pred_instance_3d["bbox_score_3d"]: float = float(score)

                pred_instances_3d.append(pred_instance_3d)
            
            non_annotated_dataset_info["data_list"][frame_index]["pred_instances_3d"] = pred_instances_3d

        else:
            # This sample was filtered out - add empty prediction
            non_annotated_dataset_info["data_list"][frame_index]["pred_instances_3d"] = []
            print(f"Sample {frame_index} was filtered out - setting empty prediction")
        
        pseudo_labeled_dataset_info["data_list"].append(non_annotated_dataset_info["data_list"][frame_index])
    
    print(f"Final pseudo_labeled_dataset_info length: {len(pseudo_labeled_dataset_info['data_list'])}")
    return pseudo_labeled_dataset_info


def inference(
    model_config: Config,
    model_checkpoint_path: str,
    non_annotated_info_file_name: str,
    batch_size: int = 1,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Args:
        model_config (Config): Config file for the model used for auto labeling.
        model_checkpoint_path (str): Path to the model checkpoint(.pth) used for auto labeling.
        non_annotated_info_file_name (str): Name of the non-annotated info file.
        batch_size (int, optional): Batch size for inference. Defaults to 1
        device (str, optional): Device to run the model on. Defaults to "cuda:0"
    Returns:
        Dict[str, Any]: inference results. This should be info file format.
    """
    # build dataloader
    model_config.test_dataloader.dataset.ann_file: str = (
        model_config.info_directory_path + non_annotated_info_file_name
    )
    model_config.test_dataloader.batch_size: int = batch_size
    dataset = Runner.build_dataloader(model_config.test_dataloader)
    
    # Get the mapping from the dataset
    original_to_filtered_map = get_filtered_data_mapping(dataset.dataset, dataset)

    # build model
    model = MODELS.build(model_config.model)
    load_checkpoint(model, model_checkpoint_path, map_location=device)
    model.to(get_device())
    model.eval()

    # predict pseudo label
    inference_results: List[Tuple[LiDARInstance3DBoxes, NDArray, NDArray]] = []
    progress_bar = ProgressBar(len(dataset))
    for frame_index, data in enumerate(dataset):
        inference_results.append(_predict_one_frame(model, data))
        progress_bar.update()

    # convert pseudo label to info file
    non_annotated_info_file_path: str = (
        model_config.test_dataloader.dataset.data_root + model_config.test_dataloader.dataset.ann_file
    )
    with open(non_annotated_info_file_path, "rb") as f:
        non_annotated_dataset_info: Dict[str, Any] = pickle.load(f)
    
    # Convert to info with full dataset mapping
    pseudo_labeled_dataset_info: Dict[str, Any] = _results_to_info(
        non_annotated_dataset_info,
        inference_results, 
        original_to_filtered_map
    )

    return pseudo_labeled_dataset_info
