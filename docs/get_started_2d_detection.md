
## 1. Set environment

- Set environment

```sh
git clone  https://github.com/tier4/autoware-ml
ln -s {path_to_dataset} data
```

```sh
├── data
│  └── nuimages
│  └── coco
├── Dockerfile
├── projects
├── README.md
└── work_dirs
```

- Build docker

```sh
docker build -t autoware-ml .
```

## 2. Prepare dataset
### 2.1. [Option] COCO dataset

- Download dataset from official website
- Run docker

### 2.2. [Option] T4 dataset

TBD

## 3. Train and evaluation

- You can use docker command for training as below.
  - See each [projects](projects) for detail command of training and evaluation.

```
docker run -it --rm --gpus '"device=1"' --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c '<command for each projects>'
```

## 4. Visualization

TBD
