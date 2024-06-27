# docker build
DOCKER_BUILDKIT=1 docker build -t autoware-ml ./
DOCKER_BUILDKIT=1 docker build -t autoware-ml ./projects/TransFusion/

# setup projects
docker run -it --rm --gpus 'all' --name autoware-ml --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python projects/TransFusion/setup.py develop'
docker run -it --rm --gpus 'all' --name autoware-ml --shm-size=64g -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml bash -c 'python projects/BEVFusion/setup.py develop'
