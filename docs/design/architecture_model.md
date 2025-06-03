# Architecture for ML model
## The type of ML model

At first, we prepare examples for Camera-LIDAR 3D detection model and T4dataset for each product.

![](/docs/fig/model_type_1.drawio.svg)

In these components, we define 5 types of models for deployment.
As you go downstream, the model becomes a tuned model for a specific vehicle and specific environment.

![](/docs/fig/model_type_2.drawio.svg)

For now, we prepare the JapanTaxi model and X2 model (which refers to the minibus product) as products, and we use the base model for other projects.

![](/docs/fig/model_type_3.drawio.svg)

In 3D detection, we deploy the following models:

- TransFusion-L (TransFusion LiDAR-only model) as product models for the JapanTaxi model and X2 model
- CenterPoint as product models for the JapanTaxi model and X2 model
- BEVFusion-L (BEVFusion LiDAR-only model) as an offline model

### 1. Pretrain model

"Pretrain model" is used for training the base model to increase generalization performance.
"Pretrain model" is basically trained by public datasets and pseudo-label datasets.
"Pretrain model" is managed by `AWML`.

### 2. Base model

"Base model" can be used for a wide range of projects.
"Base model" is based on a LiDAR-only model for 3D detection for general purposes.
"Base model" is basically fine-tuned using all of the T4dataset from the "Pretrain model".
"Base model" is managed by `AWML`.

### 3. Product model

"Product model" can be used for a product defined by reference design like XX1 (RoboTaxi) and X2 (RoboBus).
"Product model" can use specific sensor configurations for deployment.
It can be used for a sensor fusion model because the sensor configuration is fixed.
"Product model" is basically fine-tuned from the "Base model".
"Product model" is managed by `AWML`.

### 4. Project model

If the performance of the "product model" is not enough for some reason, the "Project model" can be used for specific projects.
"Project model" adapts to specific domains and is trained by pseudo-labels using the "Offline model".
"Project model" sometimes uses project-only datasets, which cannot be used for other projects for some reason.
"Project model" is not managed by `AWML` as it is just prepared as an interface from `AWML`, so the user should manage the "project model".

### 5. Offline model

"Offline model" can be used for offline processes like pseudo-labeling and cannot be used for real-time autonomous driving applications.
"Offline model" is based on a LiDAR-only model for 3D detection for generalization performance.
"Offline model" is basically trained using all datasets.
"Offline model" is managed by `AWML`.

## Model management of ML model
### Definition of model name

We manage the version to ML model.
We use `"algorithm_name + model_name/version"` to manage the ML model.
For example, we use as following.

- The base model of "TransFusion-L base/1.2".
- The base model of "BEVFusion-offline base/2.4".
- The base model of "CenterPoint-nearby base/3.1".
- The product model of "TransFusion-L japantaxi-gen1/1.2.2".
- The product model of "CenterPoint x2/1.2.1".
- The project model of "CenterPoint x2/1.2.3-shiojiri.2".

> algorithm_name

The word "algorithm_name" is based on string type like "CenterPoint", "TransFusion", and "BEVFusion".
Some algorithms have the name of modality.
For example, "BEVFusion-L" means the model of BEVFusion using LiDAR pointcloud input, and "BEVFusion-CL" means the model of BEVFusion using Camera inputs and LiDAR pointcloud inputs.
In addition, there are cases where we make a model for a specific purpose.
For example, we make "CenterPoint-offline", which is aimed to use auto labeling.
In another case, we would make "CenterPoint-nearby", which is aimed to improve detection of nearby objects such as pedestrians and bicycles.

> model_name

The word "model_name" is based on an enum type of "pretrain", "base", and the name of the product.
For now, we use the name of the product based on the vehicle name, which can be used as a reference.
For example, we use "japantaxi", "japantaxi-gen2", "x2", and "x2-gen2" for now.

> version

The word "version" uses a combination of integers based on semantic versioning.
There are some cases where the version has a string type.
See the section of "Versioning of ML model".

### Versioning of ML model

There are four ways to describe the version.

#### Pretrain model: `pretrain/{date}`

"Pretrain model" is an optional model, so you can skip this section to manage model versioning.
We prepare the pretrain model with pseudo T4dataset to increase generalization performance.
Pseudo T4dataset contains various vehicle types, sensor configurations, and kinds of LiDAR between Velodyne series and Pandar series.
We aim to adapt to various sensor configurations by using a pretrain model, which is trained by various pseudo T4dataset.

- Version definition
  - {date}: The date that the model was created.
    - We do not use versioning and manage the model by the document which describes the used config.
- Example
  - "CenterPoint pretrain/20241203": This model was trained on December 3rd, 2024.
- Criterion for version-up from `pretrain/{date}` to `pretrain/{next_date}`
  - If you make a new pretrain model, you update the pretrain model.

#### Base model: `base/X.Y`

We manage the model versioning based on the "base model".
If you want to make a new model, you should start from the "base model".

- Version definition
  - `X`: The major version for Autoware.
    - The version `X` handles the parameters related to ROS packages.
    - The change in the major version means that the developer of ROS software needs to check when to integrate the system.
    - Conversely, if the major version does not change, it can be used with the same ROS packages.
    - Major version zero (`0.Y`) is for initial development. Breaking changes are acceptable.
  - `Y`: The version of the base model. The version of the training configuration for the base model.
    - The condition includes changes in the training parameters, used dataset, and pretrain model.
- Example
  - "CenterPoint base/1.2": This model is based on version 1 config of ROS parameters and updated the used dataset twice.
- Criterion for version-up from `base/X.Y` to `base/(X+1).0`
  - If you want to change the parameters related to ROS packages, you need to update version `X`.
    - For example, the config of the detection range is used in both training parameters and ROS parameters.
    - Then, if it changes, we need to update both autoware-ml configs and ROS parameters.
- Criterion for version-up from `base/X.Y` to `base/X.(Y+1)`
  - If the model is trained and does not require changes to ROS parameters, we update version `Y` and deploy it for Autoware.
    - For example, if the dataset used for training changes, we update version `Y`.
    - If the pretrain model changes but the parameters related to ROS packages are not changed, we update version `Y`.
  - For Autoware users, if version `X` does not change, you can use the new model with the same version of the ROS package.

#### Product model: `{product_name}/X.Y.Z`

"Product model" is an optional model for deployment, so you can skip this section to manage model versioning.
If you want to increase the performance of perception for a particular product (i.e., a particular vehicle), you should prepare a "product model".

- Version definition
  - `{product_name}`: The name of the product. For now, we use japantaxi, japantaxi-gen2, x2, x2-gen2 as product names.
  - `X.Y`: The version of the base model
  - `Z`: The version of the product model
- Example
  - "CenterPoint x2-gen2/1.2.3": This model was fine-tuned from "CenterPoint base/1.2" and updated for the third time.
- Criterion for version-up from `{product_name}/X.Y.Z` to `{product_name}/X.Y.(Z+1)`
  - If we update the product dataset used for fine-tuning, we update version `Z`.

#### Project model: `{product_name}/X.Y.Z-{project_version}`

If there is an issue in the product model and we want to release a band-aid model with a pseudo dataset, then we release a project model.
The performance of the project model does not change significantly from the product model.
Note that the project model is a tentative model, and the official release is the next version of the product model, which has been annotated and retrained for the issue scene.
Because of that, Pseudo-T4dataset and the project model are not managed by autoware-ml.

- Version definition
  - `{product_name}/X.Y.Z`: The version of the product model
  - `{project_version}`: The version of the project model.
    - We use the pre-release name of [semantic versioning](https://semver.org/) as the project version. It contains the dataset information of the Pseudo-T4dataset.
    - Note that unlike pre-releases in semantic versioning, `X.Y.Z` < `X.Y.Z.{project_version}`, and `{project_version}` is a newer model version.
- Example
  - "CenterPoint x2-gen2/1.2.3-shiojiri.2": This model was fine-tuned from "CenterPoint x2-gen2/1.2.3".
- Criterion for version-up of `{product_name}/X.Y.Z-{project_version}`
  - For example, we update from "CenterPoint x2-gen2/1.2.3-shiojiri.2" to "CenterPoint x2-gen2/1.2.3-shiojiri.3"

### The strategy for fine tuning

We follow the strategy for fine-tuning as below.

![](/docs/fig/model_release.drawio.svg)

- 1. Introduce the additional input feature to the base model (This means a breaking change for the ROS package)
  - Update from `base/X.Y` to `base/(X+1).0`
  - Update from `{product_name}/X.Y.Z` to `{product_name}/(X+1).0.0`

This update includes changes related to ROS parameters, such as range parameters.
This update creates a new model from the beginning, so we need to release both the base model and product model.
If you need to update the pretrain model according to the base model update, you should retrain the pretrain model.

- 2. Update the base model by adding an additional database dataset
  - Update from `base/X.Y` to `base/X.(Y+1)`
  - Update from `{product_name}/X.Y.Z` to `{product_name}/X.(Y+1).0`

Every few months, we fine-tune the base model from the pretrain model and release the next version of the base model.
Note that we do not fine-tune the base model from itself, but from the pretrain model.

The reason we use all of the dataset is based on the strategy of a foundation model.
The base model is fine-tuned to adapt to a wide range of sensor configurations and driving areas.

Note that if we support two or more base models (e.g., base/1.1 and base/2.0), we either update all the models or deprecate the old versions.
We update all the base model versions: base/1.1 to base/1.2 & base/2.0 to base/2.1.
We also update all the product model versions depending on those models, if necessary: x2-gen2/1.1.3 to x2-gen2/1.2.0 & x2-gen2/2.0.0 to x2-gen2/2.1.0.

- 3. Start supporting a new product by fine-tuning from the base model
  - Start making the product model from `base/X.Y` to `{product_name}/X.Y.0`

For example, when we start releasing CenterPoint for the product of X2-Gen2, we fine-tune from `CenterPoint base/X.Y` and release the product model as `CenterPoint x2-gen2/X.Y.0`.

- 4. Update the product model by adding product dataset
  - Update from `{product_name}/X.Y.Z` to `{product_name}/X.(Y+1).0`

When a new annotated T4dataset is added, we release the product to fine-tune the base model.
Note that updating the product model will NOT trigger the project model, as they are temporary releases or release candidates.
Also, we do not fine-tune the product model from the product model, but from the base model.

- 5. Make project model
  - Create from `{product_name}/X.Y.Z` to `{product_name}/X.(Y+1).0-{project_name}`

If there is an issue from a particular project, we create a project model to deploy a band-aid model for that project.
Fine-tuning is done from the product model with Pseudo-T4dataset, which is created by the offline model.
For example, the project model `CenterPoint x2/1.2.3-shiojiri.1` is fine-tuned from `CenterPoint x2/1.2.3`.
Autoware-ml does not manage project models and Pseudo-T4dataset, and it is acceptable to retrain from a project model.

### The strategy for introduction of new algorithm

If you want to introduce a new algorithm, such as replacing the backbone or head, you should create new models as part of the new algorithm.
Rather than suddenly making disruptive changes to the existing model, you should start developing it as a separate model during the experimental period.
Here is an example of introducing a new backbone to TransFusion.

![](/docs/fig/new_algorithm.drawio.svg)

In major version zero (`0.Y`), initial development is done.
Breaking changes are acceptable.
The strategy provides a maintenance branch for existing model and a development branch for new algorithm.
The maintenance branch aims to provide a stable model for operation engineers.
The development branch aims to create new models with high performance.

### The strategy for release model

![](/docs/fig/model_release_2.drawio.svg)

- AWML introduces an option for generating product-level models, called the `product-release model`.
  - (step 1) Fine-tuning a base model using only the training dataset to create a product model.
  - (step 2) Evaluating the product model using standard validation and/or test data.
  - (step 3) Fine-tuning a new product-release model from the original base model using the full dataset (including train, val, and test splits), based on the best-performing configuration found in step 2.
  - As `product-release model`, we call `{algorithm-name} {product-name}/X.Y.Z-release`.
    - For example, we call "CenterPoint bus/1.1.1-release" for the product-release model of "CenterPoint bus/1.1.1".
- Design
  - In typical machine learning workflows, datasets are divided into train, validation, and test splits.
    - This separation is essential to detect overfitting, assess generalization performance (i.e., performance on unseen data), and provide a reliable basis for tuning and comparison.
    - Accordingly, only the training dataset is used for model training.
  - However, in product-level deployment, especially in domains like autonomous driving, the goal shifts toward maximizing performance in real-world environments.
    - In such contexts, data collection and annotation are often costly and time-consuming, and acquiring additional labeled data can be challenging.
  - Therefore, there is a strong motivation to utilize as much available data as possible for model training.
    - In particular, during early industrial phases where labeled data is few, if generalization performance has already been sufficiently evaluated, or if visual validation without labels is considered, it can be a viable strategy to train on the entire dataset to prioritize final model accuracy.
  - Importantly, AWML does not provide release models for the base model.
    - This is a deliberate design choice: in our fine-tuning strategy, the base model serves as the source for all downstream fine-tuning, and using a release model at this stage would risk data leakage.
