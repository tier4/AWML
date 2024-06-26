# [TODO]: make interface
GPUS='all'
CONFIG_DIR="projects/TransFusion/configs/t4dataset/"
CONFIG="transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_90m_768grid"
CONFIG_FILE=$CONFIG_DIR$CONFIG".py"
GPU_N=2

# train
# [TODO]: add overwrite config for GPU
docker run -it --rm --gpus $GPUS --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'bash tools/detection3d/dist_train.sh $CONFIG_FILE $GPU_N'

# eval
# docker run -it --rm --gpus $GPUS --name autoware-ml --shm-size=64g -d -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'bash tools/detection3d/test.py $CONFIG_FILE'

# visualization
# [TODO]: add scripts
