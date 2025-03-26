'''
AnyText: Multilingual Visual Text Generation And Editing
Paper: https://arxiv.org/abs/2311.03054
Code: https://github.com/tyxsspa/AnyText
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import sys
import os
import torch
from cldm.model import create_model,load_state_dict
import time



if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
else:
    print('Args are wrong, using default input and output path!')
    input_path = 'models/2/5-000006.ckpt'  # sd1.5
    output_path = './models/3/anytext_1.ckpt'


def load_base_model(model_path,output_path):
    model = create_model(config_path='./models_yaml/anytext_sd15.yaml')
    model.load_state_dict(load_state_dict('models/1/anytext_1.ckpt'), strict=True)
    unet_te_weights = {}
    unet_te_weights = torch.load(model_path)
    if 'state_dict' in unet_te_weights:
        unet_te_weights = unet_te_weights['state_dict']
    unet_te_keys = [i for i in unet_te_weights.keys()]
    model_state = model.state_dict()
    for key in model_state:
        if 'model.diffusion_model' in key or 'cond_stage_model.transformer.text_model' in key:
            new_key = key
            if new_key not in unet_te_weights:
                print(f'key {new_key} not found!')
            else:
                unet_te_keys.remove(new_key)
            model_state[key] = unet_te_weights[new_key]
    model.load_state_dict(model_state)
    torch.save(model.state_dict(), output_path)

load_base_model(input_path,output_path)
print('Done.')