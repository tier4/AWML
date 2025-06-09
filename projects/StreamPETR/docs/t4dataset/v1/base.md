# Deployed model for template base/v1.X
## Summary

## Release
### template base/v1.0


# Requirements
```
pip install packaging ninja==1.11.1.3
pip install -U fvcore>=0.1.5.post20221221
pip install flash-attn==2.7.3 --no-build-isolation

python tools/detection3d/create_data_t4dataset.py --root_path ./data --config /workspace/autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py --version xx1 --max_sweeps 1 --out_dir ./data/info/cameraonly/streampetr

```