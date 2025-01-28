
# Deploying and evaluating models on autoware using Webauto Evaluator

Once you have successfully generated `onnx,config` files for ROS2 testing, you can evaluate the model in the [Webauto Evaluator](https://docs.web.auto/user-manuals/evaluator/introduction) (if you have access to Webauto evaluator) by the following steps.  
1. Deploy the model in WebAuto with the following commands.
    ``` bash
      webauto ml artifact push --project-id <PROJECT-ID> \
        --package-id <PACKAGE-ID> \
        --path "$file" \
        --filename "$filename" \
        --description "$description"
    ```
    For newly trained models, use the `mlops-randd` (this is not the project-id it is the project-name, please search the project id in webauto ML Packages) project.
2. Once you have uploaded the files, go to `ML Packages`>`package-name`>   `New Release`
   select the artifacts you uplaoded in step 1 and create a new release (example: `transfusion_v0.1`).

3. Create a new branch in pilot-auto and modify the `.webauto-ci.yml` file in the following parts (example with transfusion):
    ``` yaml
      - name: asset-deploy
        user: autoware
        exec: ./.webauto-ci/main/asset-deploy/run.sh
        ml_packages:
        - name: transfusion
            release: transfusion_v0.1
        ...
        ...
        ...
      - name: PerformanceTest_perception
        type: perception
        suite:
        selector:
            match_labels:
            perception_test_type: performance
        simulator:
        deployment:
            type: container
            artifact: main
        runtime:
            type: gpu1/amd64/large
        parameters:
            sensing: "false"
            use_traffic_light_recognition: "false"
            lidar_detection_model: transfusion
        pre_tasks:
            - exec: source /home/autoware/pilot-auto.xx1/install/setup.bash && ros2 launch autoware_lidar_transfusion lidar_transfusion.launch.xml model_name:=transfusion_xx1 model_path:=/opt/autoware/mlmodels/transfusion/ model_param_path:=$(ros2 pkg prefix autoware_launch --share)/config/perception/object_recognition/detection/lidar_model/transfusion_xx1.param.yaml build_only:=true
            mounts:
                - volume: transfusion_model
        volumes:
            - name: transfusion_model
            path: /opt/autoware/mlmodels/transfusion
            preserve_original_files: true
    ```  

4. Once you create a branch with the above changes, go to `Catalogs` > `PerceptionPerformanceTest.cmn` > `Actions` > `Execute test`. Input all the necessary information including your branch commit id, and start the test.

5. If all the test cases pass, report the results in the PR. 
    It is possible that the build fails in which case either choose a stable pilot-auto version as base branch or debug the build issues manually.