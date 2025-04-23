#!/bin/bash
python eval/anytext_singleGPU.py \
        --ckpt_path "D:\\dachuan\\models\\hehe98\\wenchuang.ckpt" \
        --input_json "D:\\dachuan\\anytext\\AnyText\\benchmark\\benchmark\\wukong_word\\test1k.json" \
        --output_dir "D:\\dachuan\\anytext\\AnyText\\output" 
