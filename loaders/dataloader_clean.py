# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# License included under licenses/LICENSE_pase.

"""
Dataloder for the URGENT2 dataset, flexible to selecting specific datasets.
    Selected_set: [
        'dns5_fullband', 'vctk', 'libritts', 'commonvoice', 'mls_segments', 'ears', 
    ]

Clean: Clean speech
"""
import random
from torch.utils import data
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from typing import List, Union, Optional
from tqdm import tqdm
from utils import simulate_utils


DEFAULT_SPEECH_SET = ['dns5_fullband', 'vctk', 'libritts', 'commonvoice', 'mls_segments', 'ears']
        
        
class URGENT2Dataset(data.Dataset):
    def __init__(
        self, 
        cfg_yaml: str = 'configs/simulation_train.yaml',
        wav_len: float = 4.0,
        num_per_epoch: int = 10000,
        random_start: bool = False,
        default_fs: int = 16000,
        selected_fs: List = [16000, 22050, 24000, 32000, 44100, 48000],
        selected_set: Union[str, List[str]] = 'all',
        mode: str = 'train'):

        super().__init__()
        assert mode in ['train', 'validation']
        
        self.config = OmegaConf.load(cfg_yaml)
        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        
        self.default_fs = default_fs
        self.selected_fs = selected_fs
        self.mode = mode
        
        if selected_set == 'all':
            selected_set = DEFAULT_SPEECH_SET
        
        self.speech_database = {s_set: [] for s_set in selected_set}  # {'set_name': []}
        count = 0
        for scp in self.config.speech_scps:
            with open(scp, "r") as f:
                for line in tqdm(f, desc='Preparing speech database'):
                    uid, fs, audio_path = line.strip().split()
                    
                    if audio_path.startswith("/data/ssd5"):
                        audio_path = audio_path.replace("/data/ssd5", "/data/ssd0")
                    
                    selected_flags = [s in audio_path for s in selected_set]
                    if not any(selected_flags):
                        continue
                    if int(fs) >= self.default_fs:
                        idx = selected_flags.index(True)
                        s_set = selected_set[idx]

                        self.speech_database[s_set].append({"id": f"utt_{count}", "uid": uid, "fs": fs, "label": s_set, "path": audio_path})
                        count += 1
        
        self.valid_speech_set_nums = sum(1 for set_data in self.speech_database.values() if len(set_data) > 0)
        
        speech_database_stat = {k: len(item) for (k, item) in self.speech_database.items()}
        print(f"[{mode}] Speech:", speech_database_stat)
        self.sample_data_per_epoch(mode)
    
    def sample_data_per_epoch(self, mode='train'):
        self.speech_dic = []  # [{"uid": xxx, "fs": xxx, "path": xxx}, ...]

        
        if mode != 'train':
            random.seed(0)
            np.random.seed(0)
        
        for set_name, set_data in self.speech_database.items():
            if len(set_data) > 0:
                self.speech_dic.extend(
                    random.choices(set_data, k=self.num_per_epoch // self.valid_speech_set_nums + 1)
                )
        
        random.shuffle(self.speech_dic)

    
    def __getitem__(self, idx):
        fs = self.default_fs
        rng = np.random.default_rng(idx)
        
        speech_meta = self.speech_dic[idx]  # {'id': "xxx", "uid": xxx, "fs": xxx, "path": xxx}
        
        speech = speech_meta['path']
    
        try:
            speech_sample = simulate_utils.read_audio(speech, force_1ch=True, fs=fs)[0]
        except:
            print(speech)
                
        orig_len = speech_sample.shape[1]
        # select a segmen with a fixed duration in seconds
        if self.wav_len != 0:  # wav_len=0 means no cut or padding, use in test
            seg_len = int(self.wav_len*fs)
            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len-seg_len) if self.random_start else 0
                speech_sample = speech_sample[:, start_point: start_point + seg_len]
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                speech_sample = np.pad(speech_sample, ((0, 0), (0, pad_points)), constant_values=0)
        
        scale = rng.uniform(0.5, 0.95)
        speech_sample = speech_sample / (np.max(np.abs(speech_sample)) + 1e-9) * scale

        info_ = {'id': speech_meta["id"], 'uid': speech_meta["uid"], 'fs': fs, 'length': orig_len, 'speech': speech}
        return speech_sample.astype(np.float32), info_
    
    
    def __len__(self):
        return len(self.speech_dic)

   
 
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from omegaconf import OmegaConf
    import soundfile as sf
    
    config = OmegaConf.load('configs/cfg_train_vocoder.yaml')

    train_dataset = URGENT2Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])

    shape0 = None

    tmp_dir = '/data/hdd0/xiaobin.rong/experiments/train_samples'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm -rf {tmp_dir}/*")
    
    train_dataloader.dataset.sample_data_per_epoch()
    for step, (clean, info) in enumerate(tqdm(train_dataloader)):
        if shape0 is None:
            shape0 = clean.shape
            print(shape0)
        shape = clean.shape
        assert shape == shape0
        
        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
        if step == 1000:
            break

    valid_dataset = URGENT2Dataset(**config['validation_dataset'])
    valid_dataloader = data.DataLoader(valid_dataset, **config['validation_dataloader'])

    tmp_dir = '/data/hdd0/xiaobin.rong/experiments/valid_samples'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm -rf {tmp_dir}/*")

    shape0 = None
    
    info_scp = []
    for step, (clean, info) in enumerate(tqdm(valid_dataloader)):
        if shape0 is None:
            shape0 = clean.shape
            print(shape0)
        shape = clean.shape
        assert shape == shape0

        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
        if step == 1000:
            break
        

