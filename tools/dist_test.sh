#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29499}

echo $CHECKPOINT
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
~/paddlejob/workspace/env_run/output/lijiaming/conda2/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
