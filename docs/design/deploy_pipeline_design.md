## The design of deployment flow

The whole pipeline of deployment flow.

![](/docs/fig/model_deployment.drawio.svg)

- 1. Local PC of basic model provider

An engineer makes model data of basic model (any of pretrain model, base model, product model, offline model).
Model data has checkpoint of the model, training log, onnx file for Autoware, ROS parameter for Autoware.
If the model is used for not only `AWML` but also Autoware, we also upload to WebAuto model management system (Basically, we will upload product model).

We plan that we upload model data to S3 from local PC until MLOps system is constructed.

- 2. Local PC of project model provider

We got the issue from projects, we deploy for a model dedicated to that project.
So the engineer who makes project model upload to WebAuto model management system.
Because project models are not managed by `AWML`, he/she upload the model data only to WebAuto model management system.

- 3. S3 model registry

`AWML` manage pretrain model, base model, product model, and offline model.

- 4. WebAuto model management system

WebAuto model management system provide the model deployment.
In detail, please see [WebAuto document](https://docs.web.auto/en/user-manuals/evaluator-ml-pipeline/introduction).

- 5. autoware-ml user

`AWML` users can use S3 model registry as model zoo.
For example, the user of auto labeling can use offline model, and the user of fine-tuning can use base model.

- 6. Autoware user

Autoware user choose the model to use from WebAuto model management system.
In detail, please see [WebAuto document](https://docs.web.auto/en/user-manuals/evaluator-ml-pipeline/introduction).
