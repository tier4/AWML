# (TBD) Project adaptation

This pipeline is the example of domain adaptation with auto labeling.
`autoware-ml` do *not* manage the configuration for each experiment, so if you want to manage for each experiment, then you make new repository to manage it.
In this example, we use [CenterPoint](/projects/CenterPoint).

- [Support priority](https://github.com/tier4/autoware-ml/blob/main/docs/design/autoware_ml_design.md#support-priority): Tier C

## Get started
### 1. Prepare non-annotated T4dataset

- Prepare non-annotated T4dataset

### 2. Make pseudo T4dataset

We use [Pseudo label for T4dataset](/tools/t4dataset_pseudo_label_3d/) to make pseudo T4dataset

```sh
# command
```

### 3. Fine tuning

- Fine tuning by pseudo label

```sh
# command
```
