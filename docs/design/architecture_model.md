# Architecture for ML model
## The type of ML model

At first, we prepare example for Camera-LIDAR 3D detection model and T4dataset for each product.

![](/docs/fig/model_type_1.drawio.svg)

In these component, we define 5 types model for deploy.
As you go downstream, the model becomes tuned model to a specific vehicle and specific environment.

![](/docs/fig/model_type_2.drawio.svg)

For now, we prepare for RoboTaxi (as we call XX1) and RoboBus (As we call X2) as products and  we use base model directory for other projects.
In 3D detection, we deploy TransFusion-L (TransFusion LiDAR-only model) and CenterPoint as product model (XX1 model and X2 model), and BEVFusion-L (BEVFusion LiDAR-only model) as offline model.

![](/docs/fig/model_type_3.drawio.svg)

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

## Management of ML model
### ML model versioning

We use semantic version to ML model versioning.
We use "algorithm name + model name + version" to manage the ML model.
For example, we use as following.

- We release the base model of "TransFusion-L T4base v0.2.0".
- We release the product model of "TransFusion-L T4XX1 v0.2.0".
- We release the project model of "TransFusion-L T4X2-50m v0.1.3".
- We release the project model of "CenterPoint T4X2 v0.1.3".

> Algorithm name

"Algorithm name" is like "CenterPoint", "TransFusion", and "BEVFusion".
Some algorithm name the kind of modality.
For example, "BEVFusion-L" means the model of BEVFusion using LiDAR pointcloud input and "BEVFusion-CL" means the model of BEVFusion using Camera inputs and LiDAR pointcloud inputs.

> Model name

Model name choose in "T4pretrain", "T4base", product name (For now, we use T4XX1 and T4X2).
As optional name, when we prepare multiple models like 90m model and 120m, we name the model name including it.
For example, we could use "TransFusion-L T4X2" (default model) and "TransFusion-L T4X2-50m" (50m model) to use for various projects.

> version

We use the model version of X.Y.Z as following.

- Major version (X): The version of parameters related to ROS packages
  - If we need to change ROS parameters, we update the major version of ML model.
  - The changing of major version means that the developer of ROS software need to check when to integrate for the system.
     - Conversely, if the major version is not changed, it can be used for same ROS packages.
  - For example, the config of the detection range is used in both training parameter and ROS parameters. Then, if it is changed, we need to update both `autoware-ml` configs and ROS parameters.
- Minor version (Y): The version of training configuration
  - If the model is trained and it doesn't need the change of ROS parameters (it means that the ML model can be used for same version of ROS package), we update the minor version.
  - The condition include
    - Change the training parameters
    - Change using dataset
    - Change using pretrain model
- Patch version (Z): The version of patch level fine-tuning
  - If fine tuned by pseudo T4dataset, we update the patch version.
  - Update of patch version means performance does not change significantly.
  - Pseudo T4dataset and project model are not managed by autoware-ml.

### Fine tuning strategy

We follow the strategy for fine tuning as below.

![](/docs/fig/model_release.drawio.svg)

At first we prepare the pretrain model with pseudo T4dataset to increase generalization performance.
Pseudo T4dataset contains various vehicle type, sensor configuration, and kinds of LiDAR between Velodyne series and Pandar series.
We are to adapt various sensor configuration by using pretrain model, which is trained by various pseudo T4dataset.

After that, we train base model with all of annotated T4dataset based on pretrain model.
The reason why We use all of the dataset is based on the strategy of foundation model.
The base model is fine-tuned to adapt a wide range of sensor configuration and driving area.

When new annotated T4dataset is added, we release new base model at first.
After that we release the product model using the base model as pre-trained model.
If some problem like domain-specific objects, we release the project model by retraining the product model using pseudo label with the offline model.
Note that we do not retrain the product model and project model from the model of before version because it is difficult to trace the model.
