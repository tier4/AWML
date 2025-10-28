# Auto Labeling 3D Pipeline

`launch.py` launches `auto_labeling_3d` in a single end-to-end workflow:

1. **download_checkpoint** – Download model checkpoints from Model Zoo URLs
2. **create_info_data** – Create info files for saving pseudo labels from a non-annotated T4Dataset
3. **ensemble_infos** – Ensemble/filter pseudo labels across one or more models
4. **attach_tracking_id** – Assign consistent tracking ID to every 3D bounding box across frames per scene
5. **create_pseudo_t4dataset** – Write processed pseudo labels to a T4Dataset
6. **change_directory_structure** – Change the directory structure from non-annotated T4Dataset format to pseudo-label T4Dataset format

## Quick start

```python
python tools/auto_labeling_3d/entrypoint/launch.py tools/auto_labeling_3d/entrypoint/configs/example.yaml
```
