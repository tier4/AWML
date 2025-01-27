# Add new algorithm

If you want to release new algorithm in a project, you may add/fix config files in `projects/{model_name}/configs/*.py`.

You can change the backbone or the head in the model in some case.
If you change some breaking changes in the model, you should change the algorithm name.
For example, you should change from "CenterPoint" to "CenterPoint-ConvNetXT" when changing the backbone of CenterPoint.

When you make a PR, you should write

- Why do you want to update the model?
- How do you evaluate the model? (e.g. based on mAP for Odaiba data, AP for pedestrians, etc)

For PR review list with code owner

- [ ] Write the experiment results
- [ ] Write results of training and evaluation including analysis for the model in PR.
- [ ] Upload the model and logs
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)

## Evaluate the model
### Use new algorithm for Autoware

If you use new algorithm for Autoware, you should evaluate not only performance but also inference time and memory to use.
Here is example.

```
- Note:
  - Time (A), Memory (A)
    - Environment: [Autoware 0.40.0](https://github.com/autowarefoundation/autoware/releases/tag/0.40.0)
  - Time (P), Memory (P)
    - Environment: pytorch

| Model                                | mAP         | Time (A) | Memory (A) | Time (P) | Memory (P) |
| ------------------------------------ | ----------- | -------- | ---------- | -------- | ---------- |
| CenterPoint base/1.0 (120m)          | 64.4        | 18.2 ms  | 4.7 GB     | 471 ms   | 0.96 GB    |
| CenterPoint-ConvNeXt base/0.2 (120m) | 65.3 (+0.9) | 30.4 ms  | 5.6 GB     | 691 ms   | 1.75 GB    |
| CenterPoint-ConvNeXt base/0.1 (120m) | 68.6 (+4.2) | 48.1 ms  | 5.9 GB     | 1008 ms  | 3.01 GB    |
```

### Use new algorithm for active learning

If you use new algorithm for active learning, here is example.

```
- Note:
  - Time (P), Memory (P)
    - Environment: pytorch

| Model                                | mAP         | Time (P) | Memory (P) |
| ------------------------------------ | ----------- | -------- | ---------- |
| CenterPoint base/1.0 (120m)          | 64.4        | 471 ms   | 0.96 GB    |
| CenterPoint-ConvNeXt base/0.2 (120m) | 65.3 (+0.9) | 691 ms   | 1.75 GB    |
| CenterPoint-ConvNeXt base/0.1 (120m) | 68.6 (+4.2) | 1008 ms  | 3.01 GB    |
```
