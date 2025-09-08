# Architecture for evaluation pipeline
**Note that this work is still in progress and undergoing significant development, please check the [Release Plans](#Release-plans) for upcoming changes.**

An evaluation pipeline in `AWML` is a systematic framework designed to assess the performance of a trained model. It involves a series of well-defined steps that help determine how well a model generalizes to unseen data. An effective evaluation pipeline ensures that the model is robust, reliable, and well-tuned for actual deployment, while minimizing the risk of overfitting or underfitting. This design doc outlines the main design of evaluation pipeline in both experimental and deployment environments.


# Dependencies
- [autoware_perception_evaluation](https://github.com/tier4/autoware_perception_evaluation/blob/develop/docs/en/perception/design.md)
    - Repository to evaluate perception/sensing tasks

# Design
## High-level design (HLD)
Figure below shows the top-level overview of evaluation pipeline in AWML:

![](/docs/fig/awml_evaluation_architecture.drawio.svg)

## Low-level design (LLD)
### `<class> T4MetricV2(...)`
  - Note that `T4MetricV2` is a class name during development, and it will eventually replace `T4Metric` after a transition period
  - A class to execute evaluation pipeline for a trained machine learning model
	- Parameters of `T4MetricV2` are as follows:
		| Arguments                            | Type                         | Description                                                                                   							|
		| :----------------------------------- | :--------------------------  | :---------------------------------------------------------------------------------------------------------- |
		| `data_root`                      		 | `str`                        | Dataset path                                                                                 								|
		| `ann_file`                       		 | `str`                        | Pickle file with annotations                                                                  							|
		| `test_mode`                       	 | `bool`                       | Set true for test mode, otherwise, validation mode                                                          |
		| `evaluation_config`              		 | `Dict[str, Any]`             | Parameters of `PerceptionEvaluationManager`                                                   							|
		| `critical_object_filter_configs` 		 | `Dict[str, Any]`             | Parameters of `PerceptionEvaluationManager.add_frame_result`                                  							|
		| `perception_pass_fail_configs`   		 | `Dict[str, Any]`             | Parameters of `PerceptionEvaluationManager.add_frame_result`                                  						  |
		| `confidence_score_thresholds_path`   | `Optional[str]`              | Set to load confidence score threadolds from the path, otherwise it will calibrate from aggregated metrics  |
  - Difference of test mode and validation mode is that confidence scores will be calibrated in the validation mode, but test mode will use the confidence scores from the validation mode

  - `evaluation_config`
	  - Example of default configs (e.g., `autoware_ml/configs/detection3d/dataset/t4dataset/base.py`)
		```python
		evaluation_configs=dict(
			evaluation_task="detection",
			target_labels=class_names,
			ignore_attributes=ignore_attributes,
			max_x_position=102.4,
			max_y_position=102.4,
			min_point_numbers=[5, 5, 5, 5, 5],
			label_prefix="autoware",
			merge_similar_labels=False,
			allow_matching_unknown=False,
			plane_distance_thresholds=[0.5, 1.0, 2.0, 3.0],
			iou_3d_thresholds=[0.5],
		)
		critical_object_filter_config=dict()
		perception_pass_fail_config=dict()
		```
	- Example of test evaluator in config.py:
		```python
		evaluation_configs = dict(
				max_x_position=121.0,
				max_y_position=121.0,
		)
		test_evaluator = dict(
				type="T4MetricV2",
				data_root=data_root,
				test_mode=True,
				ann_file=data_root + info_directory_path + _base_.info_test_file_name,
				evaluation_configs=evaluation_configs,
				critical_object_filter_config=__base__.critical_object_filter_config,
				perception_pass_fail_config=__base.__perception_pass_fail_config,
				confidence_score_thresholds_path=None
		)
		val_evaluator = dict(
				type="T4MetricV2",
				data_root=data_root,
				test_mode=False,
				ann_file=data_root + info_directory_path + _base_.info_test_file_name,
				evaluation_configs=evaluation_configs,
				critical_object_filter_config=__base__.critical_object_filter_config,
				perception_pass_fail_config=__base.__perception_pass_fail_config,
				confidence_score_thresholds_path=None
		)
		```

	- Prediction
		- Predictions/GTs from every frame are converted to `PerceptionFrameResult`, and grouped by the same scenario id
		- For every scenerio id, it saves a dict of `PerceptionFrameResult` with a frame id, which can be represented by `sample_idx`
		- `results.pkl` save results with the following template:
			```python
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
			```

	- Metric computation
	  - For every frame, it calls `autoware_perception_eval.PerceptionEvaluationManager` to compute metrics, and save them to `scene_metrics.json`
	  - `scene_metrics.json` save metrics from every scene with the following template:
		```python
		{
			'scene_0': {
				'frame_0': {
					'all': {
						'metric_1': {
							'car':,
							...
						},
						'metric_2'; {
							'car':,
							...
						},
						...
						}
				},
				'frame_1': {
					'all': {
						'metric_1': {
							'car':,
							...
						},
						'metric_2'; {
							'car':,
							...
						},
						...
				},
				...
			},
			'scene_1': {
				'frame_0': {
					'all': {
						'metric_1': {
							'car':,
							...
						},
						'metric_2'; {
							'car':,
							...
						},
						...
				},
				'frame_1': {
					'all': {
						'metric_1': {
							'car':,
							...
						},
						'metric_2'; {
							'car':,
							...
						},
						...
				},
				...
			},
		}
		```
	  - Aggregate metrics and calibrate confidence scores, and save them in `aggregated_metrics.json` with the following template:
		```python
		{
			'aggregated_metrics': {
				{
					"metrics": {
							"mAP_{metric1}": ,
							"mAPH_{metric1}": ,
							"mAP_{metric2}": ,
							"mAPH_{metric2}":
					},
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
		```

### `<class> T4MetricVisualization(...)`
  - A class to render metric results, for example, PR curve and confusion metrics from an evaluation
	- Parameters of `T4MetricVisalization` are as follows:
		| Arguments                            | Type                         | Description                                                                                   							|
		| :----------------------------------- | :--------------------------  | :---------------------------------------------------------------------------------------------------------- |
		| `data_root`                      		 | `str`                        | Dataset path                                                                                 								|
		| `ann_file`                       		 | `str`                        | Pickle file with annotations                                                                  							|
		| `scene_metric_path`             		 | `str`                        | Json path with metric results for every scene                                                             	|
		| `aggregated_metric_path`        		 | `str`                        | Json path with aggregated metric results                                                             				|
		| `top_k_worst`        		             | `Optional[int]`              | Set to render top-k worst scenarios in the evaluation pipeline                                              |

## Release plans
- autoware_perception_evaluation:
    - [ ] Implementation of nuScene metrics in autoware_perception_evaluation, this includes NDS and calibration of confidence thresholds
    - [ ] Make filter optional
    - [ ] Support loading FrameGroundTruth and sensor data without providing dataset_paths
- AWML:
    - [ ] Integrate PerceptionFrameResult and refactor inference to save predictions/gts in every step, also save intermediate results results.pickle for all scenes
    - [ ] Configuration of autoware_perception_evaluation through experiment configs, and process T4Frame with autoware_perception_evaluation.add_frame_result and autoware_perception_evaluation.get_scene_result
    - [ ] Visualize metrics and worst K samples (`T4MetricVisualization`)
    - [ ] Unit tests for simple cases
- Misc:
    - [ ] Resample train/val/test splits
