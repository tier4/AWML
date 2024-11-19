# [TODO]: make interface
GPUS='all'
CONFIG_DIR="projects/TransFusion/configs/t4dataset/"
CONFIG="transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_90m_768grid"
CONFIG_FILE=$CONFIG_DIR$CONFIG".py"
GPU_N=2

# train
# [TODO]: add overwrite config for GPU
# docker run -it --rm --gpus $GPUS --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'bash tools/detection3d/dist_train.sh $CONFIG_FILE $GPU_N'

# eval
# docker run -it --rm --gpus $GPUS --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'bash tools/detection3d/test.py $CONFIG_FILE'

# onnx deploy
# DIR="work_dirs/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1" && \
# python tools/detection3d/deploy.py projects/TransFusion/configs/deploy/transfusion_lidar_tensorrt_dynamic-20x5.py $DIR/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1.py $DIR/epoch_50.pth data/t4dataset/database_v1_1/0171a378-bf91-420e-9206-d047f6d1139a/0/data/LIDAR_CONCAT/0.pcd.bin --device cuda:0 --work-dir /workspace/$DIR/onnx
# DIR="work_dirs/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid-t4xx1" && \
# python projects/TransFusion/scripts/fix_graph.py $DIR/onnx/end2end.onnx

# deploy ros parameter
