# Architecture for ML model
## The type of ML model

At first, we prepare example for Camera-LIDAR 3D detection model and T4dataset for each product.

![](/docs/fig/model_type_1.drawio.svg)

In these component, we define 5 types model for deploy.
As you go downstream, the model becomes tuned model to a specific vehicle and specific environment.

![](/docs/fig/model_type_2.drawio.svg)

For now, we prepare for JapanTaxi model and X2 model (it means minibus product) as products and we use base model for other projects.

![](/docs/fig/model_type_3.drawio.svg)

In 3D detection, we deploy

- TransFusion-L (TransFusion LiDAR-only model) and CenterPoint as product model as JapanTaxi model and X2 model
- BEVFusion-L (BEVFusion LiDAR-only model) as offline model

### 1. Pretrain model

"Pretrain model" is used for training base model to increase generalization performance.
"Pretrain model" is basically trained by public dataset and pseudo label dataset.
"Pretrain model" is managed by `autoware-ml`.

### 2. Base model

"Base model" can be used for a wide range of projects.
"Base model" is based on LiDAR-only model for 3D detection to general purpose.
"Base model" is basically fine-tuned by all of T4dataset from "Pretrain model".
"Base model" is managed by `autoware-ml`.

### 3. Product model

"Product model" can be used for a product defined by reference design like XX1 (RoboTaxi) and X2 (RoboBus).
"Product model" can use specific sensor configuration to deploy.
It can be used for sensor fusion model because sensor configuration is fixed.
"Product model" is basically fine-tuned from "Base model".
"Product model" is managed by `autoware-ml`.

### 4. Project model

If the performance "product model" is not enough in some reason, "Project model" can be used for specific project.
"Project model" adapts to specific domain, trained by pseudo label using "Offline model".
"Project model" sometimes uses for project-only dataset, which cannot use for other project for some reason.
"Project model" is not managed by `autoware-ml`  as it is just prepared as interface from `autoware-ml`, so the user should manage "project model".

### 5. Offline model

"Offline model" can be used to offline process like pseudo label and cannot be used for real-time autonomous driving application.
"Offline model" is based on LiDAR-only model for 3D detection for generalization performance.
"Offline model" is basically trained by all dataset.
"Offline model" is managed by `autoware-ml`.

## Model management of ML model

We manage the version to ML model.
We use `"algorithm_name + model_name/version"` to manage the ML model.
For example, we use as following.

- The base model of "TransFusion-L base/1.2".
- The base model of "CenterPoint-offline base/2.4".
- The base model of "CenterPoint-nearby base/3.1".
- The product model of "TransFusion-L japantaxi-gen1/1.2.2".
- The project model of "TransFusion-L-50m base/1.2".
- The project model of "CenterPoint x2/1.2.3-shiojiri.2".

> algorithm_name

The word of "algorithm_name" is based on string type like "CenterPoint", "TransFusion", and "BEVFusion".
Some algorithm has the name of modality.
For example, "BEVFusion-L" means the model of BEVFusion using LiDAR pointcloud input and "BEVFusion-CL" means the model of BEVFusion using Camera inputs and LiDAR pointcloud inputs.
In addition to it, there are cases where we make a model for a specific purpose.
For example, we make "CenterPoint-offline", which is aimed to use auto labeling.
In another case, we would make "CenterPoint-nearby", which is aimed to improve to detect the near object of pedestrians and bicycles.

> model_name

The word of "model name" is based on enum type of "pretrain", "base", the name of product.
For now, we use the name of product based on the vehicle name which can be used as reference.
For example, we use "japantaxi", "japantaxi-gen2","x2", and "x2-gen2" for now.

> version

The word of "version" use a combination of integers based on semantic versioning.
There are some cases that version has string type.
See the section of "Versioning of ML model"

### Versioning of ML model

There are four ways to describe the version.

#### Pretrain model: `pretrain/{date}`

"Pretrain model" is optional model, so you can skip this section to manage model versioning.
We prepare the pretrain model with pseudo T4dataset to increase generalization performance.
Pseudo T4dataset contains various vehicle type, sensor configuration, and kinds of LiDAR between Velodyne series and Pandar series.
We are to adapt various sensor configuration by using pretrain model, which is trained by various pseudo T4dataset.

- Version definition
  - {date}: The date to make the model.
    - We do not use the versioning and manage the model by the document which write the used config.
- Example
  - "CenterPoint pretrain/20241203": This model trained at 3rd Dec. 2024
- Criterion for version-up from `pretrain/{date}` to `pretrain/{next_date}`
  - If you make new pretrain model, you update the pretrain model.

#### Base model: `base/X.Y`

We manage the model versioning based on "base model".
If you want to make new model, you should start from "base model".

- Version definition
  - `X`: The major version for Autoware.
    - The version `X` handle the parameters related to ROS packages
    - The changing of major version means that the developer of ROS software need to check when to integrate for the system.
    - Conversely, if the major version is not changed, it can be used for same ROS packages.
  - `Y`: The version of base model. The version of training configuration for base model.
    - The condition include change of the training parameters, using dataset, and pretrain model.
- Example
  - "CenterPoint base/1.2": This model is based on version 1 config of ROS parameter and update used dataset twice.
- Criterion for version-up from `base/X.Y` to `base/(X+1).0`
  - If you want to change the parameters related to ROS package like, you need to update the version `X`.
    - For example, the config of the detection range is used in both training parameter and ROS parameters.
    - Then, if it is changed, we need to update both autoware-ml configs and ROS parameters.
- Criterion for version-up from `base/X.Y` to `base/X.(Y+1)`
  - If the model is trained and it doesn't need the change of ROS parameters, we update the version `Y` and deploy for Autoware.
    - For example, if the dataset using for training is changing, we update the version `Y`.
    - If the pretrain model is changing but the parameter related to ROS parameters is not changing, we update the version `Y`.
  - For Autoware user, if the version of `X` does not change, you can use the new model for same version of ROS package.

#### Product model: `{product_name}/X.Y.Z`

"Product model" is optional model for deployment, so you can skip this section to manage model versioning.
If you want to increase the performance of perception for a particular product (it means a particular vehicle), you should prepare "product model".

- Version definition
  - `{product_name}`: The name of product. For now we use japantaxi, japantaxi-gen2, x2, x2-gen2 as product_name.
  - `X.Y`: The version of base model
  - `Z`: The version of product model
- Example
  - "CenterPoint x2-gen2/1.2.3": This model do fine-tuning from "CenterPoint base/1.2" and update third times.
- Criterion for version-up from `{product_name}/X.Y.Z` to `{product_name}/X.Y.(Z+1)`
  - We update the product dataset to use for fine tuning, we update the version `Z`

#### Project model: `{product_name}/X.Y.Z-{project_version}`

If there are some issue in product model and we want to release band-aid model with pseudo dataset, then we release project model.
The performance of project model does not change significantly from product model.
Note that project model is tentative model and the official release is the next version of the product model that the issue scene has been annotated and retrained.
Because of that, Pseudo-T4dataset and project model are not managed by autoware-ml.

- Version definition
  - `{product_name}/X.Y.Z`: The version of product model
  - `{project_version}`: The version of project model.
    - We use pre-release name of [semantic versioning](https://semver.org/) as project version. It contains the dataset information of Pseudo-T4dataset.
    - Note that unlike pre-release in semantic version, `X.Y.Z` < `X.Y.Z.{project_version}`, and `{project_version}` is a newer model version.
- Example
  - "CenterPoint x2-gen2/1.2.3-shiojiri.2": This model do fine-tuning from "CenterPoint x2-gen2/1.2.3"
- Criterion for version-up of `{product_name}/X.Y.Z-{project_version}`
  - For example, we update from "CenterPoint x2-gen2/1.2.3-shiojiri.2" to "CenterPoint x2-gen2/1.2.3-shiojiri.3"

### Fine tuning strategy

We follow the strategy for fine tuning as below.

![](/docs/fig/model_release.drawio.svg)

- 1. Introduce the additional input feature to the base model (It means breaking change for ROS package)
  - Update from `base/X.Y` to `base/(X+1).0`
  - Update from `{product_name}/X.Y.Z` to `{product_name}/(X+1).0.0`

This update include the update related to ROS parameter like range parameter.
This update make new model from beginning, so we need to release from base model, and product model.
If you need to update pretrain model according to update of base model, you retrain the pretrain model.

- 2. Update the base model by adding additional database dataset
  - Update from `base/X.Y` to `base/X.(Y+1)`
  - Make from `{product_name}/X.Y.Z` to `{product_name}/X.(Y+1).0`

Once every few months, we do fine-tuning base model from pretrain model and release next version of base model.
Note that we do not do fine-tuning base model from base model but pretrain model.

The reason why We use all of the dataset is based on the strategy of foundation model.
The base model is fine-tuned to adapt a wide range of sensor configuration and driving area.

Note that if we support two or more base models e.g. base/1.1 and base/2.0, update all the models or either deprecate the old versions.
Update all the base model versions: base/1.1 to base/1.2 & base/2.0 to base/2.1.
Update all the product model versions depending on those models if necessary: x2-gen2/1.1.3 to x2-gen2/1.2.0 & x2-gen2/2.0.0 to x2-gen2/2.1.0

- 3. Start supporting a new product by fine-tuning from the base model
  - Start to make product model from `base/X.Y` to `{product_name}/X.Y.0`

For example, when we start to release CenterPoint for the product of X2-Gen2, we do fine-tuning from `CenterPoint base/X.Y` and release the product model as `CenterPoint x2-gen2/X.Y.0`.

- 4. Update the product model by adding product dataset
  - Update from `{product_name}/X.Y.Z` to `{product_name}/X.(Y+1).0`

When new annotated T4dataset is added, we release the product to do fine-tuning the base model.
Note that the update of the product model will NOT trigger the project model, since they are temporary release / release candidates.
Note that we do not do fine-tuning product model from product model but base model.

- 5. Make project model
  - Make from `{product_name}/X.Y.Z` to `{product_name}/X.(Y+1).0-{project_name}`

By getting the issue from a particular project, we make a project model to deploy band-aid model for the project.
Fine-tuned is done from the product model with Pseudo-T4dataset, which is made by offline model.
For example, the project model `CenterPoint x2/1.2.3-shiojiri.1` is fine-tuned from `CenterPoint x2/1.2.3`.
Autoware-ml do not manage project model and Pseudo-T4dataset and it is OK to re-train from project model.
