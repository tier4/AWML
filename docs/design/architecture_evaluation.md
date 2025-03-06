# Architecture for evaluation pipeline
This document outlines the main design of evaluation pipeline in AWML. Note that part of the work is still ongoing, and please check the [Release Plans](#Release-plans) for upcoming changes.

## Design
Figure below shows the top-level overview of evaluation pipeline in AWML:

![](/docs/fig/awml_evaluation_architecture.drawio.svg)

### Implementation

- Input:
    - `t4dataset_infos_test.pkl` is an example of info pickle that generated as an input file to mmdetectio3d-based framework
- Inference:
    - Input: `t4dataset_infos_test.pkl`
		- Output: `results.pkl`
    - Predictions/GTs from every frame are converted to `PerceptionFrameResult`, and grouped by the same scenario id
		- For every scenerio id, it saves a dict of `PerceptionFrameResult` with `FrameId`, which can be represented by `sample_idx`
		- `results.pkl` should save results in the following template:
		  {
				'scene_0': {
					'frame_0': PerceptionFrameResult(predictions, gt_boxes),
					'frame_1': PerceptionFrameResult(predictions, gt_boxes),
					...
				},
				'scene_1': {
					'frame_0': PerceptionFrameResult(predictions, gt_boxes),
					...
				},
				...
			}

- Metric computation:
	- Input: `results.pkl` and `experiment_config.py`
	- Output: `scene_metrics.json` and `aggregated_metrics.json`
	- Build `autoware_perception_eval.PerceptionEvaluationManager`, and call `autoware_perception_eval.PerceptionEvaluationManager` to compute metrics for every frame
	- Save metrics from every scene in the following template:
	{
		'scene_0': {
			'frame_0': {
				'car': {
					'metric_1': ,
					'metric_2': ,
					... 
				},
				'bicycle': {
					'metric_1': ,
					'metric_2': ,
					... 
				}, 
				...
			},
			'frame_1': {
				'car': {
					'metric_1': ,
					'metric_2': ,
					... 
				},
				'bicycle': {
					'metric_1': ,
					'metric_2': ,
					... 
				}, 
				...
			},
			...
		}, 
		'scene_1': {
			'frame_0': {
				'car': {
					'metric_1': ,
					'metric_2': ,
					... 
				},
				'bicycle': {
					'metric_1': ,
					'metric_2': ,
					... 
				}, 
				...
			}
		}, 
	}
	- Aggregate metrics and calibrate confidence scores, and save them in `aggregated_metrics.json` with the following template:
	{
		'aggregated_metrics': {
			{
				'car': {
					'metric_1': ,
					'metric_2': ,
					'optimal_confidence_threshold':
					... 
				},
				'bicycle': {
					'metric_1': ,
					'metric_2': ,
					'optimal_confidence_threshold':
					...
				}
			}
		},
	}

- Visualization:
  - Input: `metrics.json` and `scene_metrics.json`
	- Output: List of figures about metrics, for example, PR curvers/visualization of top-k worst scenes/confusion matrix
	- Render results of evaluation

## Release plans
- autoware_perception_evaluation:
    - [] Implementation of nuScene metrics in autoware_perception_evaluation, this includes NDS and calibration of confidence thresholds
    - [] Make filter optional
    - [] Support loading FrameGroundTruth and sensor data without providing dataset_paths
- AWML:
    - [] Integrate PerceptionFrameResult and refactor inference to save predictions/gts in every step, also save intermediate results results.pickle for all scenes
    - [] Configuration of autoware_perception_evaluation through experiment configs, and process T4Frame with autoware_perception_evaluation.add_frame_result and autoware_perception_evaluation.get_scene_result
    - [] Visualize metrics and worst K samples
    - [] Unit tests for simple cases
- Misc:
    - [] Resample train/val/test splits 