<div align="center">
<h1>LitePT: Lighter Yet Stronger Point Transformer</h1>

[**Yuanwen Yue**](https://ywyue.github.io/), [**Damien Robert**](https://drprojects.github.io/), [**Jianyuan Wang**](https://jytime.github.io/),
[**Sunghwan Hong**](https://sunghwanhong.github.io/), [**Jan Dirk Wegner**](https://dm3l.uzh.ch/wegner/group-leader)<br>[**Christian Rupprecht**](https://chrirupp.github.io/),
[**Konrad Schindler**](https://igp.ethz.ch/personen/person-detail.html?persid=143986)

ETH Zurich,
University of Oxford,
University of Zurich

<a href="https://arxiv.org/abs/2512.13689"><img src="https://img.shields.io/badge/arXiv-LitePT-red" alt="Paper PDF"></a>
<a href="https://litept.github.io/"><img src="https://img.shields.io/badge/Project_Page-LitePT-green" alt="Project Page"></a>
<a href='https://huggingface.co/prs-eth/LitePT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>

<img
  src="https://raw.githubusercontent.com/LitePT/LitePT.github.io/main/assets/figures/teaser.png"
  alt="teaser"
  width="800"
/>

</div>

**LitePT** is a lightweight, high-performance 3D point cloud architecture.
<u>Left</u>: LitePT-S has $3.6\times$ fewer parameters, $2\times$ faster runtime and $2\times$ lower memory footprint than PTv3. Moreover, it remains faster and more memory-efficient even when scaled up to LitePT-L with a parameter count twice that of PTv3. <u>Right</u>: Already the smallest variant, LitePT-S, matches or outperforms state-of-the-art point cloud backbones across a range of benchmarks.

## News

- **2025-12-16:** Paper, project page, code, models are released.

## Preparation

### Environment

- Create an environment and install pytorch and other required packages:

  ```shell
  git clone https://github.com/prs-eth/LitePT.git
  cd LitePT
  uv sync
  ```

- Additional requirements.
  - For **instance segmentation**:
    ```shell
    uv sync --extra insseg
    ```

### Use LitePT in Your Own Project

To simplify integrating LitePT into your own projects, we provide a standalone implementation of LitePT. To include LitePT in your codebase, copy the directories `LitePT/litept` and `LitePT/libs/pointrope` into your project, arranging them as shown below:

```
Your_project/
└── .../
    └── LitePT/
        ├── litept/
        └── libs/pointrope
```

We also provide a `demo_use.py` script that illustrates how to use the standalone LitePT with an example point cloud input.

### Data

We prepare and organize the data in the same way with Pointcept [Data Preparation](https://github.com/Pointcept/Pointcept#data-preparation). All data should be placed in `LitePT/data`.

## Model Zoo

### Semantic segmentation

| Model    | Params |  Benchmark   | val mIoU |                                                Config                                                |                                                       Checkpoint                                                       |
| :------- | -----: | :----------: | :------: | :--------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| LitePT-S |  12.7M |   NuScenes   |   82.2   |   [link](https://github.com/prs-eth/LitePT/blob/main/configs/nuscenes/semseg-litept-small-v1m1.py)   |   [Download](https://huggingface.co/prs-eth/LitePT/blob/main/nuscenes-semseg-litept-small-v1m1/model/model_best.pth)   |
| LitePT-S |  12.7M |    Waymo     |   73.1   |    [link](https://github.com/prs-eth/LitePT/blob/main/configs/waymo/semseg-litept-small-v1m1.py)     |    [Download](https://huggingface.co/prs-eth/LitePT/blob/main/waymo-semseg-litept-small-v1m1/model/model_best.pth)     |
| LitePT-S |  12.7M |   ScanNet    |   76.5   |   [link](https://github.com/prs-eth/LitePT/blob/main/configs/scannet/semseg-litept-small-v1m1.py)    |   [Download](https://huggingface.co/prs-eth/LitePT/blob/main/scannet-semseg-litept-small-v1m1/model/model_best.pth)    |
| LitePT-S |  12.7M | Structured3D |   83.6   | [link](https://github.com/prs-eth/LitePT/blob/main/configs/structured3d/semseg-litept-small-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/structured3d-semseg-litept-small-v1m1/model/model_best.pth) |
| LitePT-B |  45.1M | Structured3D |   85.1   | [link](https://github.com/prs-eth/LitePT/blob/main/configs/structured3d/semseg-litept-base-v1m1.py)  | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/structured3d-semseg-litept-base-v1m1/model/model_best.pth)  |
| LitePT-L |  85.9M | Structured3D |   85.4   | [link](https://github.com/prs-eth/LitePT/blob/main/configs/structured3d/semseg-litept-large-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/structured3d-semseg-litept-large-v1m1/model/model_best.pth) |

### Instance segmentation

| Model      | Params | Benchmark  | mAP<sub>25</sub> | mAP<sub>50</sub> | mAP  |                                               Config                                               |                                                      Checkpoint                                                      |
| :--------- | -----: | :--------: | :--------------: | :--------------: | :--: | :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| LitePT-S\* |  16.0M |  ScanNet   |       78.5       |       64.9       | 41.7 |  [link](https://github.com/prs-eth/LitePT/blob/main/configs/scannet/insseg-litept-small-v1m2.py)   |  [Download](https://huggingface.co/prs-eth/LitePT/blob/main/scannet-insseg-litept-small-v1m2/model/model_best.pth)   |
| LitePT-S\* |  16.0M | ScanNet200 |       40.3       |       33.1       | 22.2 | [link](https://github.com/prs-eth/LitePT/blob/main/configs/scannet200/insseg-litept-small-v1m2.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/scannet200-insseg-litept-small-v1m2/model/model_best.pth) |

### Object detection

| Model  | Params | Benchmark | mAPH | Config |                                                   Checkpoint                                                    |
| :----- | -----: | :-------: | :--: | :----: | :-------------------------------------------------------------------------------------------------------------: |
| LitePT |   9.0M |   Waymo   | 70.7 |  link  | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/waymo-objdet-litept-small-v1m3/model/model_best.pth) |

## Training

### Semantic segmentation

If you want to resume training with your checkpoint, specify `weight=<path/to/your_checkpoint.pth>`:

```shell
### NuScenes + LitePT-S
python3 -m tools.train --config-file ./configs/nuscenes/semseg-litept-small-v1m1.py --num-gpus 4 \
  --options \
  weight=<path/to/nuscenes/model_best.pth>
```

## Testing

### Semantic segmentation

```shell
### NuScenes + LitePT-S
python -m tools.test --config-file ./configs/nuscenes/semseg-litept-small-v1m1.py --num-gpus 1 \
  --options \
  weight=<path/to/nuscenes/model_best.pth> show=True
```

## Deployment

### Semantic segmentation

```shell
### NuScenes + LitePT-S
python -m tools.torch2onnx --config-file ./configs/nuscenes/semseg-litept-small-v1m1.py --num-gpus 1 \
  --options \
  weight=<path/to/nuscenes/model_best.pth>
```

## Checklist

- [x] Release models, code for semantic segmentation and instance segmentation.
- [ ] Release object detection code.

## Acknowledgment

Thanks to these great repositories: [Pointcept](https://github.com/Pointcept/Pointcept), [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3), [Superpoint Transformer](https://github.com/drprojects/superpoint_transformer), [CroCo](https://github.com/naver/croco), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Citation

If you find this project useful, please cite:

```
@article{yuelitept2025,
    title={{LitePT: Lighter Yet Stronger Point Transformer}},
    author={Yue, Yuanwen and Robert, Damien and Wang, Jianyuan and Hong, Sunghwan and Wegner, Jan Dirk and Rupprecht, Christian and Schindler, Konrad},
    journal={arXiv preprint arXiv:2512.13689},
    year={2025}
}
```
