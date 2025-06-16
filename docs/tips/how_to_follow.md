## How to follow model update

Here, we explain how to track the differences on ML model update.
We look at PRs that include model updates, such as this one: https://github.com/tier4/AWML/pull/56.

Changes related to the dataset are in `autoware_ml/configs/t4dataset/*.yaml`.
If you can access right to T4dataset, instead of asking “What’s the difference between v1.6 and v1.7?”, it’s more accurate to ask “What kind of data is in `autoware_ml/configs/t4dataset/*.yaml`?”
Similarly, if you use own customized T4dataset, you can follow the changes in the data used for model training and evaluation by tracking changes in the dataset configuration.

Experimental conditions are defined in `projects/CenterPoint/config/`, so check for changes there to see if the model or training parameters have been modified.
In this case, you’ll notice that the `pts_voxel_encoder` part of the model has changed.
(If you want to understand the model in detail, it’s probably best to read the implementation directly.)
