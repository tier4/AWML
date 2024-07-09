# Architecture for whole ML pipeline
## Whole data pipeline for ML model with Autoware

TBD

## Whole architecture for ML model with Autoware

TBD

## ML model deployment pipeline

We define 4 types model for deploy.

Here is example for Camera-LIDAR 3D detection model for deploy pipeline.

![](/docs/fig/model_pipeline.drawio.svg)

### 1. Base model

"Base model" can be used for a wide range of projects.
"Base model" is based on LiDAR-only model for 3D detection to general purpose.
"Base model" is basically trained by all dataset.

### 2. Product model

"Product model" can be used for a product defined by reference design like XX1 and X2.
"Product model" can use specific sensor configuration to deploy for sensor fusion model.
"Product model" is basically trained by all product dataset.

### 3. Project model

"Project model" can be used for specific project.
"Project model" adapts to specific domain, trained by pseudo label using "Offline model".
"Project model" sometimes uses for project-only dataset, which cannot use for other project for some reason.

### 4. Offline model

"Offline model" can be used to offline process like pseudo label and cannot be used for real-time autonomous driving application.
"Offline model" is based on LiDAR-only model for 3D detection for generalization performance.
"Base model" is basically trained by all dataset.
