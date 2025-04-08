#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
python -m pytorch_fid \
    /root/autodl-fs/wukong-40k \
    /root/autodl-tmp/text_edit_train
    