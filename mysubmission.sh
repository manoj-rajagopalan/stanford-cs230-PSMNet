#!/bin/bash

PYTHONPATH=${PYTHONPATH}:models:utils \
python -m debugpy --listen 5678 \
    submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath /cs230-datasets/kitti2015/testing/ \
                     --loadmodel ./output/checkpoint_9.tar
