#!/bin/bash

PYTHONPATH=${PYTHONPATH}:models \
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath dataset/ \
               --epochs 10 \
               --savemodel ./output



# python finetune.py --maxdisp 192 \
#                    --model stackhourglass \
#                    --datatype 2015 \
#                    --datapath dataset/data_scene_flow_2015/training/ \
#                    --epochs 300 \
#                    --loadmodel ./trained/checkpoint_10.tar \
#                    --savemodel ./trained/
