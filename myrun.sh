#!/bin/bash

PYTHONPATH=${PYTHONPATH}:models \
python main.py \
    --tag 01-train_bs_8-crop_w_256-no_pretrain \
    --comment "Reduced training batch size to 8 (was 12). Cropped image width to 256 (Was 512). No pre-trained model." \
    --maxdisp 192 \
    --model stackhourglass \
    --datapath ./dataset/ \
    --epochs 10 \
    --savemodel ./output/



# python finetune.py --maxdisp 192 \
#                    --model stackhourglass \
#                    --datatype 2015 \
#                    --datapath dataset/data_scene_flow_2015/training/ \
#                    --epochs 300 \
#                    --loadmodel ./trained/checkpoint_10.tar \
#                    --savemodel ./trained/
