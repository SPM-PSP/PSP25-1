#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python eval/eval_dgocr.py \
        --img_dir /root/autodl-tmp/text_edit_train \
        --input_json /root/autodl-tmp/test1k.json