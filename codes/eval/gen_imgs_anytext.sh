#!/bin/bash
python eval/anytext_singleGPU.py \
        --ckpt_path /root/autodl-tmp/anytext_origin.ckpt \
        --input_json /root/autodl-tmp/test1k.json \
        --output_dir /root/autodl-tmp/text_edit_train \
        --input_dir /root/autodl-tmp/imgs