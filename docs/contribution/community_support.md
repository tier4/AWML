# Community support

This page shows community supports abround `AWML`.
`AWML` is based on [Autoware Core & Universe strategy](https://autoware.org/autoware-overview/).
We hope that this page promote the community between Autoware and ML researchers and engineers.

## Papers
### About `AWML`

- The arXiv paper of AWML: https://arxiv.org/abs/2506.00645
  - This paper contains the whole design of AWML.

```
@misc{tanaka2025awmlopensourcemlbasedrobotics,
      title={AWML: An Open-Source ML-based Robotics Perception Framework to Deploy for ROS-based Autonomous Driving Software},
      author={Satoshi Tanaka and Samrat Thapa and Kok Seang Tan and Amadeusz Szymko and Lobos Kenzo and Koji Minoda and Shintaro Tomie and Kotaro Uetake and Guolong Zhang and Isamu Yamashita and Takamasa Horibe},
      year={2025},
      eprint={2506.00645},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00645},
}
```

- The arXiv paper for domain adaptation between different sensor configuration: https://arxiv.org/abs/2509.04711.

![](/docs/fig/finetuning.drawio.svg)

```
@misc{tanaka2025domainadaptationdifferentsensor,
      title={Domain Adaptation for Different Sensor Configurations in 3D Object Detection},
      author={Satoshi Tanaka and Kok Seang Tan and Isamu Yamashita},
      year={2025},
      eprint={2509.04711},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.04711},
}
```

- The arXiv paper of experimental evaluation system: https://arxiv.org/abs/2507.00190

```
@misc{tanaka2025rethink3dobjectdetection,
      title={Rethink 3D Object Detection from Physical World},
      author={Satoshi Tanaka and Koji Minoda and Fumiya Watanabe and Takamasa Horibe},
      year={2025},
      eprint={2507.00190},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.00190},
}
```

## Libraries
### AWMLPrediction

- AWMLPrediction

We are implementing `AWMLPrediction` for now, which aims to deploy ML-based prediction model for Autoware.

## Tools
### mm-project-template

- [mm-project-template](https://github.com/scepter914/mm-project-template)

This repository is the project template based on [mm series](https://github.com/open-mmlab).
You can start from this template and you can add code of `/tools/*` and `/projects/*` from `AWML` to use for your a new algorithm or a new tool.

## Models
