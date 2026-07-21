# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# License included under licenses/LICENSE_pase.
"""
Dataloder for the URGENT2 dataset, flexible to selecting specific datasets.
    Selected_set: [
        'dns5_fullband', 'vctk', 'libritts', 'commonvoice', 'mls_segments', 'ears', 
    ]

Noisy: Speech added noise and reverberation, and other distortions (with a specific probability)
Clean: Clean speech
"""
import random
from torch.utils import data
import numpy as np
import soundfile as sf
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig
from collections import defaultdict
from typing import List, Union, Optional
from tqdm import tqdm
from utils import simulate_utils


DEFAULT_SPEECH_SET = ['dns5_fullband', 'vctk', 'libritts', 'commonvoice', 'mls_segments', 'ears']

DEFAULT_NOISE_SET =['dns5_fullband', 'wham_noise_48k', 'fsd50k', 'fma', 'wind']

DEFAULT_RIR_SET = ['dns5_fullband']

        
class URGENT3Dataset(data.Dataset):
    def __init__(
        self, 
        cfg_yaml: str = 'configs/simulation_train.yaml',
        wav_len: float = 4.0,
        num_per_epoch: int = 10000,
        random_start: bool = False,
        snr_range: List = [-5, 15],
        default_fs: int = 16000,
        selected_fs: List = [16000, 22050, 24000, 32000, 44100, 48000],
        selected_set: Union[str, List[str]] = 'all',
        mode: str = 'train'):

        super().__init__()
        assert mode in ['train', 'validation']
        # assert selected_set == 'all'
        
        self.config = OmegaConf.load(cfg_yaml)
        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        
        self.snr_range = snr_range
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
                        
        self.noise_database = {n_set: [] for n_set in DEFAULT_NOISE_SET}
        for scp in self.config.noise_scps:
            with open(scp, "r") as f:
                for line in tqdm(f, desc='Preparing noise databse'):
                    uid, fs, audio_path = line.strip().split()
                    selected_flags = [s in audio_path for s in DEFAULT_NOISE_SET]
                    if any(selected_flags):
                        idx = selected_flags.index(True)
                        n_set = DEFAULT_NOISE_SET[idx]
                        self.noise_database[n_set].append({"uid": uid, "fs": fs, "label": n_set, "path": audio_path})
        
        for scp in self.config.wind_noise_scps:
            with open(scp, "r") as f:
                for line in tqdm(f, desc='Preparing wind noise database'):
                    uid, fs, audio_path = line.strip().split()
                    self.noise_database['wind'].append({"uid": uid, "fs": fs, "label": "wind", "path": audio_path})

        self.rir_database = {r_set: [] for r_set in DEFAULT_RIR_SET}
        for scp in self.config.rir_scps:
            with open(scp, "r") as f:
                for line in f:
                    uid, fs, audio_path = line.strip().split()
                    selected_flags = [s in audio_path for s in DEFAULT_RIR_SET]
                    if any(selected_flags):
                        idx = selected_flags.index(True)
                        r_set = DEFAULT_RIR_SET[idx]
                        self.rir_database[r_set].append({"uid": uid, "fs": fs, "label": r_set, "path": audio_path})
        
        self.valid_speech_set_nums = sum(1 for set_data in self.speech_database.values() if len(set_data) > 0)
        self.valid_noise_set_nums = sum(1 for set_data in self.noise_database.values() if len(set_data) > 0)
        self.valid_rir_set_nums = sum(1 for set_data in self.rir_database.values() if len(set_data) > 0)
        
        # print(f"[{mode}] Number of speech:", sum([len(s_set) for s_set in self.speech_database.values()]))
        # print(f"[{mode}] Number of noise:", sum([len(n_set) for n_set in self.noise_database.values()]))
        # print(f"[{mode}] Number of rir:", sum([len(r_set) for r_set in self.rir_database.values()]))
        speech_database_stat = {k: len(item) for (k, item) in self.speech_database.items()}
        noise_database_stat = {k: len(item) for (k, item) in self.noise_database.items()}
        rir_database_stat = {k: len(item) for (k, item) in self.rir_database.items()}
        print(f"[{mode}] Speech:", speech_database_stat)
        print(f"[{mode}] Noise:", noise_database_stat)
        print(f"[{mode}] RIR:", rir_database_stat)

        self.sample_data_per_epoch(mode)

    def sample_data_per_epoch(self, mode='train'):
        self.speech_dic = []  # [{"uid": xxx, "fs": xxx, "path": xxx}, ...]
        self.noise_dic = defaultdict(list)  # {'set_name': [], ...}
        self.rir_dic = defaultdict(list)  # {'set_name': [], ...}
        
        if mode != 'train':
            random.seed(0)
            np.random.seed(0)
        
        for set_name, set_data in self.speech_database.items():
            if len(set_data) > 0:
                self.speech_dic.extend(
                    random.choices(set_data, k=self.num_per_epoch // self.valid_speech_set_nums + 1)
                )
        
        for set_name, set_data in self.noise_database.items():
            if len(set_data) > 0:
                self.noise_dic[set_name].extend(
                    random.choices(set_data, k=self.num_per_epoch // self.valid_noise_set_nums + 1)
                )

        for set_name, set_data in self.rir_database.items():
            if len(set_data) > 0:
                self.rir_dic[set_name].extend(
                    random.choices(set_data, k=self.num_per_epoch // self.valid_rir_set_nums + 1)
                )
        
        random.shuffle(self.speech_dic)
        
        self.augmentations = list(self.config.augmentations.keys())
        weight_augmentations = [v["weight"] for v in self.config.augmentations.values()]
        self.weight_augmentations = weight_augmentations / np.sum(weight_augmentations)
                    
    
    def __getitem__(self, idx):
        fs = self.default_fs
        rng = np.random.default_rng(idx)
        
        speech_meta = self.speech_dic[idx]  # {"uid": xxx, "fs": xxx, "path": xxx}
        idx = speech_meta["id"]
        uid = speech_meta["uid"]
        
        if rng.random() < self.config.prob_reverberation:
            rkey = "none"
            rir = "none"
        else:
            rkey = random.choice(list(self.rir_dic.keys()))
            rir_meta = random.choice(self.rir_dic[rkey])
            rir = rir_meta['path']
        
        nkey = random.choice(list(self.noise_dic.keys())) 
        noise_meta = random.choice(self.noise_dic[nkey])
        nuid = noise_meta["uid"]
        snr = rng.integers(*self.snr_range, endpoint=True)
        
        speech = speech_meta['path']
        noise = noise_meta['path']
    
        try:
            speech_sample = simulate_utils.read_audio(speech, force_1ch=True, fs=fs)[0]
        except:
            print(speech)
        try:
            noise_info = sf.info(noise)
            noise_fs = noise_info.samplerate
            noise_length = int(noise_info.duration * noise_fs)
        except:
            print(noise)
        
        if noise_length > noise_fs * 10:
            start = rng.integers(0, noise_length-noise_fs*10)
            stop = start + 10*noise_fs
            noise_sample = simulate_utils.read_audio(noise, force_1ch=True, fs=fs, start=start, stop=stop)[0]
        else:
            noise_sample = simulate_utils.read_audio(noise, force_1ch=True, fs=fs)[0]
                
        orig_len = speech_sample.shape[1]
                
        if rir != "none":
            rir_sample = simulate_utils.read_audio(rir, force_1ch=True, fs=fs)[0]
            noisy_speech = simulate_utils.add_reverberation(speech_sample, rir_sample)
            # make sure the clean speech contains early reflections
            early_rir_sample = simulate_utils.estimate_early_rir(rir_sample, fs=fs, early_rir_ms=0.05)
            speech_sample = simulate_utils.add_reverberation(speech_sample, early_rir_sample)
        else:
            noisy_speech = deepcopy(speech_sample)
            
        # augmentations
        use_wind_noise = np.random.random() < self.config.prob_wind_noise

        num_aug = np.random.choice(
            range(len(self.config.num_augmentations)),
            p=self.config.num_augmentations,
        )
        if num_aug == 0 or (rir != "none" and rkey == 'HighRev'):
            aug = ["none"]
        else:
            aug = np.random.choice(
                self.augmentations,
                p=self.weight_augmentations,
                size=num_aug,
                replace=False,
            )
            # As wind-noise simulation include clipping,
            # we exclude clipping from augmentation list
            while use_wind_noise and "clipping" in aug:
                aug = np.random.choice(
                    self.augmentations,
                    p=self.weight_augmentations,
                    size=num_aug,
                    replace=False,
                )

        # simulation with non-linear wind-noise mixing
        if nuid.startswith("wind_noise"):
            wn_conf = self.config.wind_noise_config
            threshold_ = np.random.uniform(*wn_conf["threshold"])
            ratio_ = np.random.uniform(*wn_conf["ratio"])
            attack_ = np.random.uniform(*wn_conf["attack"])
            release_ = np.random.uniform(*wn_conf["release"])
            sc_gain_ = np.random.uniform(*wn_conf["sc_gain"])
            clipping_threshold_ = np.random.uniform(*wn_conf["clipping_threshold"])
            clipping_ = np.random.random() < wn_conf["clipping_chance"]
            
            try:
                noisy_speech, noise_sample = simulate_utils.wind_noise(
                    noisy_speech,
                    noise_sample,
                    fs,
                    uid,
                    float(threshold_),
                    float(ratio_),
                    float(attack_),
                    float(release_),
                    float(sc_gain_),
                    bool(clipping_),
                    float(clipping_threshold_),
                    float(snr),
                    rng=rng,
                )
            except:
                pass
        # just an additive noise
        else:
            noisy_speech, noise_sample = simulate_utils.mix_noise(
                noisy_speech, noise_sample, snr=snr, rng=rng
            )
        
        # select a segmen with a fixed duration in seconds
        if self.wav_len != 0:  # wav_len=0 means no cut or padding, use in test
            seg_len = int(self.wav_len*fs)
            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len-seg_len) if self.random_start else 0
                noisy_speech = noisy_speech[:, start_point: start_point + seg_len]
                speech_sample = speech_sample[:, start_point: start_point + seg_len]
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                noisy_speech = np.pad(noisy_speech, ((0, 0), (0, pad_points)), constant_values=0)
                speech_sample = np.pad(speech_sample, ((0, 0), (0, pad_points)), constant_values=0)

        # apply an additional augmentation
        for augmentation in aug:
            if augmentation == "none" or augmentation == "":
                pass
            elif augmentation.startswith("wind_noise"):
                pass
            elif augmentation.startswith("bandwidth_limitation"):
                res_type, fs_new = simulate_utils.gen_bandwidth_limitation_params(fs=fs, res_type="random")
                fs_new = 8000
                noisy_speech = simulate_utils.bandwidth_limitation(
                    noisy_speech, fs=fs, fs_new=int(fs_new), res_type=res_type
                )
            elif augmentation.startswith("clipping"):
                this_aug = self.config.augmentations[augmentation]
                min_ = np.random.uniform(*this_aug["clipping_min_quantile"])
                max_ = np.random.uniform(*this_aug["clipping_max_quantile"])
                noisy_speech = simulate_utils.clipping(noisy_speech, min_quantile=min_, max_quantile=max_)
                
            elif augmentation.startswith("codec"):
                this_aug = self.config.augmentations[augmentation]
                codec_config = random.choice(this_aug["config"])
                format, encoder, qscale = (
                    codec_config["format"],
                    codec_config["encoder"],
                    codec_config["qscale"],
                )
                
                if encoder is not None and isinstance(encoder, ListConfig):
                    encoder = random.choice(encoder)
                if qscale is not None and isinstance(qscale, ListConfig):
                    qscale = np.random.randint(*qscale)
                qscale = max(int(qscale), 1)
                noisy_speech = simulate_utils.codec_compression(
                    noisy_speech, fs, format=format, encoder=encoder, qscale=qscale
                )

            elif augmentation.startswith("packet_loss"):
                this_aug = self.config.augmentations[augmentation]
                packet_duration_ms_ = this_aug["packet_duration_ms"]
                packet_loss_indices_ = simulate_utils.gen_packet_loss_params(seg_len, fs, packet_duration_ms_,
                    this_aug["packet_loss_rate"],
                    this_aug["max_continuous_packet_loss"], rng)
                noisy_speech = simulate_utils.packet_loss(
                    noisy_speech, fs, packet_loss_indices_, int(packet_duration_ms_)
                )
            else:
                raise NotImplementedError(augmentation)
        
        scale = np.random.uniform(0.5, 0.95)
        noisy_speech = noisy_speech / (np.max(np.abs(noisy_speech)) + 1e-9) * scale
        speech_sample = speech_sample / (np.max(np.abs(speech_sample)) + 1e-9) * scale

        info_ = {'id': idx, 'uid': uid, 'fs': fs, 'length': orig_len, 'speech': speech, 'noise': noise, 'rir': rir, 'snr': snr}

        return noisy_speech.astype(np.float32), speech_sample.astype(np.float32), info_
    
    
    def __len__(self):
        return len(self.speech_dic)

   
 
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from omegaconf import OmegaConf
    import soundfile as sf
    
    config = OmegaConf.load('configs/cfg_train_dewavlm.yaml')

    train_dataset = URGENT3Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])

    shape0 = None

    tmp_dir = '/data/hdd0/xiaobin.rong/experiments/train_samples'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm -rf {tmp_dir}/*")
    
    train_dataloader.dataset.sample_data_per_epoch()
    
    f = open('/data/hdd0/xiaobin.rong/experiments/train_samples/_info.scp', 'w')
    for step, (noisy, clean, info) in enumerate(tqdm(train_dataloader)):
        if shape0 is None:
            shape0 = noisy.shape
            print(shape0)
        shape = noisy.shape
        assert shape == shape0
        
        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_noisy.wav", noisy[0].numpy().squeeze(), int(info['fs'][0]))
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
            f.write(f"{info['id'][0]} {info['speech'][0]} {info['noise'][0]} {info['rir'][0]} {info['snr'][0]}\n")
        if step == 200:
            break
    f.close()
    
    valid_dataset = URGENT3Dataset(**config['validation_dataset'])
    valid_dataloader = data.DataLoader(valid_dataset, **config['validation_dataloader'])

    tmp_dir = '/data/hdd0/xiaobin.rong/experiments/valid_samples'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm -rf {tmp_dir}/*")

    shape0 = None
    
    info_scp = []
    f = open('/data/hdd0/xiaobin.rong/experiments/valid_samples/_info.scp', 'w')
    for step, (noisy, clean, info) in enumerate(tqdm(valid_dataloader)):
        if shape0 is None:
            shape0 = noisy.shape
            print(shape0)
        shape = noisy.shape
        assert shape == shape0

        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_noisy.wav", noisy[0].numpy().squeeze(), int(info['fs'][0]))
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
            f.write(f"{info['id'][0]} {info['speech'][0]} {info['noise'][0]} {info['rir'][0]} {info['snr'][0]}\n")
    
        if step == 1000:
            break
        
    f.close()
