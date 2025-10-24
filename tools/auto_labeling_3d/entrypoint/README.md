# Auto Labeling 3D Pipeline

`launch.py` launches `auto_labeling_3d` in a single end-to-end workflow:

1. **create_info_data** – Create info files for saving pseudo labels from a non-annotated T4Dataset
2. **ensemble_infos** – Ensemble/filter pseudo labels across one or more models
3. **attach_tracking_id** – Assign consistent tracking ID to every 3D bounding box across frames per scene
4. **create_pseudo_t4dataset** – Write processed pseudo labels to a T4Dataset

## Quick start

```python
python tools/auto_labeling_3d/entrypoint/launch.py tools/auto_labeling_3d/entrypoint/configs/example.yaml
```
