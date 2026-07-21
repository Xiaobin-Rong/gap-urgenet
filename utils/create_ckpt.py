# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Adapted under Apache-2.0
# Source: https://github.com/cisco-open/pase/blob/main/pase/utils/create_ckpt.py
# License included under licenses/LICENSE_pase.

import torch
from omegaconf import OmegaConf


cfg_path =  "/data/hdd0/xiaobin.rong/experiments/challenge/exp_urgent3/exp_postnet_2025-11-04-15h13m/config.yaml"
ckpt_path = "/data/hdd0/xiaobin.rong/experiments/challenge/exp_urgent3/exp_postnet_2025-11-04-15h13m/checkpoints/model_090.tar"
save_path = "/data/hdd0/xiaobin.rong/pretrained/GAP-URGENet/PostNet.pt"

config = OmegaConf.load(cfg_path)
config =  OmegaConf.to_container(config)
state_dict = torch.load(ckpt_path, map_location='cpu')

# print(state_dict.keys())
# exit()
new_dict = {}

if 'generator' in state_dict.keys():
    model_dict = state_dict['generator']
elif 'model' in state_dict.keys():
    model_dict = state_dict['model']
else:
    raise ValueError("Keys mismatch!")

new_dict['model'] = model_dict
new_dict['cfg'] = config['postnet_config']
# new_dict['info'] = config

print(new_dict.keys())

torch.save(new_dict, save_path)

