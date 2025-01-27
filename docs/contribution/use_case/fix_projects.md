# Fix code in `/projects`

You can fix code in a project more casually than fixing codes with `autoware_ml/*` because the area of ​​influence is small.
However, if the model is used for Autoware and if you want to change a model architecture, you need to check deploying to onnx and running at ROS environment.

For PR review list with code owner for the project

- [ ] Write the log of result for the trained model
- [ ] Upload the model and logs
- [ ] Update documentation for the model
- [ ] Check deploying to onnx file and running at Autoware environment (If the model is used for Autoware and you change model architecture)
