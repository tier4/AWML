#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
TASK=$3
CHECKPOINT=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ "$TASK" = "train" ]; then
	python -m torch.distributed.launch \
			--nnodes=$NNODES \
			--node_rank=$NODE_RANK \
			--master_addr=$MASTER_ADDR \
			--nproc_per_node=$GPUS \
			--master_port=$PORT \
			$(dirname "$0")/train.py \
			$CONFIG \
			--launcher pytorch ${@:5}
elif [ "$TASK" = "test" ]; then
	python -m torch.distributed.launch \
			--nnodes=$NNODES \
			--node_rank=$NODE_RANK \
			--master_addr=$MASTER_ADDR \
			--nproc_per_node=$GPUS \
			--master_port=$PORT \
			$(dirname "$0")/test.py \
			$CONFIG \
			$CHECKPOINT \
			--launcher pytorch ${@:5}
else
    echo "Invalid TASK: $TASK"
    echo "Usage: $0 <CONFIG> <GPUS> <train|test> [additional args]"
    exit 1
fi
