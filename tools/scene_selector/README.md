# scene selector

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier S

## config

- [Object number threshold scene selector with 2D detection](configs/det2d_object_num_selector/)
- [(TBD) Object number threshold scene selector with 2D open vocabulary](configs/open_vocab_2d_object_num_selector/)
- [(TBD) Select scene with VLM QA](configs/vlm_qa_selector/)
- [(TBD) rosbag selector](configs/rosbag/)

## Get started
### 1. setup environment

TBD

### 2. Select scene

- [choice] Select scene from pictures
  - TBD

```sh
python tools/scene_selector/image_selector.py {config_file} {directory or image_file}
```

- [choice] Select scene from rosbag
  - TBD

```sh
python tools/select_scene/rosbag_selector.py {config_file} {rosbag_config_file} {rosbag_file} --visualization
```
