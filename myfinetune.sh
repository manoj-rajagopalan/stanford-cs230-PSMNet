#! /bin/bash

PYTHONPATH=${PYTHONPATH}:models:utils \
python finetune.py --maxdisp 192 \
       	--model stackhourglass \
       	--datatype 2015 \
       	--datapath /cs230-datasets/KITTI-2015-Manoj/training/ \
       	--epochs 300 \
	    --loadmodel ./output/checkpoint_9.tar \
       	--savemodel ./output/finetuned/
