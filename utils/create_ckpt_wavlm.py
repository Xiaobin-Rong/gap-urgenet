# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Adapted under Apache-2.0
# Source: https://github.com/cisco-open/pase/blob/main/pase/utils/create_ckpt_wavlm.py
# License included under licenses/LICENSE_pase.

import sys
sys.path.append("..")
import torch
from models.wavlm.feature_extractor_plc import WavLM_feat

model = WavLM_feat()
ckpt_finetuned = "/data/hdd0/xiaobin.rong/experiments/challenge/exp_urgent3/exp_dewavlm-omni_2025-09-27-23h05m/checkpoints/model_190.tar"
checkpoint = torch.load(ckpt_finetuned, map_location="cpu")

model.load_state_dict(checkpoint['model'])

state_dict = {
    "cfg": model.wavlm.cfg.__dict__,
    "model": model.wavlm.state_dict(),
        
    }

save_path = "/data/hdd0/xiaobin.rong/pretrained/UniPASE"
wavlm_ckpt_path_new = f"{save_path}/DeWavLM-Omni.pt"
torch.save(state_dict, wavlm_ckpt_path_new)