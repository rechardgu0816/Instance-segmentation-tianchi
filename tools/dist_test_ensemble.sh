#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_ensemble.py $CONFIG \
    --checkpoint   work_dirs/cascade_htc_mixup_mul/epoch_11.pth\
                   work_dirs/cascade_htc_mixup_mul/epoch_12.pth \
                   work_dirs/cascade_htc_mixup_mul/epoch_10.pth \
    --format-only \
    --options "jsonfile_prefix=./ensemble_test_results"
    --launcher pytorch ${@:4}
