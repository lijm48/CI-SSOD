#!/usr/bin/env bash
set -x

TYPE=$1
FOLD=$2
PERCENT=$3
GPUS=$4
PORT=${PORT:-29500}
shift 4

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'baseline' ]]; then
    python  -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/soft_teacher/ubsoft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} $@
else
    python  -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py configs/soft_teacher/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k.py --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} $@
fi

# python  -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py configs/soft_teacher/ubsoft_teacher_grs_faster_rcnn_r50_caffe_fpn_coco_180k.py --launcher pytorch \
#     --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5} 