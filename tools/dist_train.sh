#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
# PORT=${PORT:-29346}

PORT=${PORT:-29500}
RAND=$[$RANDOM % 99]
PORT=$[PORT+RAND]
INCREMENT=23

isfree=$(netstat -tapln | grep $PORT)

while [[ -n "$isfree" ]]; do
  echo "PORT: $PORT is not free, trying a new one"
  PORT=$[PORT+INCREMENT]
  isfree=$(netstat -tapln | grep $PORT)  
done

echo "PORT: $PORT is free, continue with code: "

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
