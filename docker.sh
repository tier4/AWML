DOCKER_BUILDKIT=1 docker build -t awml-ros2 ./tools/setting_environment/ros2/  --build-arg CACHEBUST=$(date +%s)
docker run --rm -it --gpus all \
  --mount type=bind,src=/mnt/qnapdata,dst=/data\
  --mount type=bind,src=/mnt/qnapdata,dst=/mnt/qnapdata \
  --ipc=host \
  -e CUBLAS_WORKSPACE_CONFIG=:16:8 \
  -v "$PWD":/workspaces/AWML\
  -w /workspaces/AWML \
  -p 5000:5000 \
  awml-ros2 \
  bash -c "nohup mlflow server --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 & exec bash"
