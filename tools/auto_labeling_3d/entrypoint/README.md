# Auto Labeling 3D Pipeline

`launch.py` stitches the existing Auto Labeling 3D utilities into a single end-to-end workflow:

1. **create_info_data** – build pseudo-labelled info files from a non-annotated dataset
2. **ensemble_infos** – ensemble/filter pseudo labels across one or more models
3. **attach_tracking_id** – assign consistent tracking IDs per scene
4. **create_pseudo_t4dataset** – write final pseudo labels in T4Dataset format

## Quick start

```python
python tools/auto_labeling_3d/entripoint/launch.py tools/auto_labeling_3d/entripoint/configs/example.yaml
```
