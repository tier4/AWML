# deploy_detection3d

This pipeline can use for deployment of detection3d models.

## 1. Set environment

```sh
git clone https://github.com/tier4/autoware-ml
ln -s {path_to_dataset} data
```

- Run script

```sh
./pipeline/deploy_detection3d/scripts/init_environment.sh
```

## 2. Prepare dataset

- Download new dataset by using [download_t4dataset](/tools/download_t4dataset/README.md).
- Update info file

```sh
# XX1
docker run -it --rm --gpus 'all' --name autoware-ml --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py --version xx1 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name'
# X2
docker run -it --rm --gpus 'all' --name autoware-ml --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python tools/detection3d/create_data_t4dataset.py --root_path ./data/t4dataset --config autoware_ml/configs/detection3d/dataset/t4dataset/x2.py --version x2 --max_sweeps 2 --out_dir ./data/t4dataset/info/user_name'
```

## 4. Train

TBD

## 5. Check results

TBD
