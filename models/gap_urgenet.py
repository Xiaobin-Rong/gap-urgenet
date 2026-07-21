# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Adapted under Apache-2.0
# Source: https://github.com/cisco-open/pase/blob/main/models/pase.py
# License included under licenses/LICENSE_pase.

import torch
import torch.nn as nn
from .predictor.tfgridnet import TFGridNet as Predictor
from .wavlm.feature_extractor_plc import WavLM_feat as Encoder
from .adapter.vocos.adapter import VocosAdapter as Adapter
from .vocoder.vocos.vocoder import VocosVocoder as Decoder
from .postnet.tfgrid_cws import TFGridNet as PostNet
from torchaudio.functional import resample


class GAP_URGENet(nn.Module):
    def __init__(
        self, 
        predictor_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/GAP-URGENet/Predictor.pt",
        dewavlm_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/GAP-URGENet/DeWavLM-Omni.pt",
        adapter_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/GAP-URGENet/Adapter.pt",
        vocoder_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/GAP-URGENet/Vocoder.pt",
        postnet_ckpt_path="/data/hdd0/xiaobin.rong/pretrained/GAP-URGENet/PostNet.pt",
    ):
        super().__init__()
        self.encoder = Encoder(dewavlm_ckpt_path, output_layer=[1,24])
        self.adapter = Adapter.from_pretrained(adapter_ckpt_path)
        self.decoder = Decoder.from_pretrained(vocoder_ckpt_path)
        
        self.predictor = Predictor.from_pretrained(predictor_ckpt_path)
        self.postnet = PostNet.from_pretrained(postnet_ckpt_path)

    @torch.no_grad()
    def forward(self, x, sr_in, sr_out=None, enable_plc=True):
        """
        Args:
            x (torch.Tensor): noisy speech with shape of (B, L) or (B, 1, L)
            sr_in (int): sampling rate of input speech
            sr_out (int): sampling rate of output speech
            enable_plc (bool): whether to perform PLC
        Return:
            y (torch.Tensor): enhanced speech with shape of (B, L).
        """
        n_samples = x.shape[-1]
        
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, L)
        
        # -------------------
        # Generative branch
        # -------------------
        noisy_feat_a, enh_feat_p = self.encoder(x, sr=sr_in, mask=enable_plc)
        enh_feat_a = self.adapter(noisy_feat_a, enh_feat_p)
        enh_gen_16k = self.decoder(enh_feat_a.transpose(1, 2)).squeeze(1)  # (B, T)
        enh_gen_48k = resample(enh_gen_16k, orig_freq=16000, new_freq=48000)
        
        # -------------------
        # Predictive branch
        # -------------------
        if sr_in != 16000:
            x = resample(x, orig_freq=sr_in, new_freq=16000)
        enh_pred_16k = self.predictor(x, sr=16000)
        enh_pred_48k = resample(enh_pred_16k, orig_freq=16000, new_freq=48000)
            
        # -------------------
        # PostNet fusion
        # -------------------
        if enh_gen_48k.shape[-1] < enh_pred_48k.shape[-1]:
            enh_gen_48k = torch.nn.functional.pad(enh_gen_48k, (0, enh_pred_48k.shape[-1]-enh_gen_48k.shape[-1]))
        else:
            enh_gen_48k = enh_gen_48k[..., :enh_pred_48k.shape[-1]]
            
        enh_2ch = torch.stack([enh_pred_48k, enh_gen_48k], dim=1)
        enh_fuse = self.postnet(enh_2ch, sr_in=48000)
        
        # -------------------
        # Resample
        # -------------------
        if sr_out is None:
            sr_out = sr_in
        
        y = resample(enh_fuse, orig_freq=48000, new_freq=sr_out)
        
        if sr_out == sr_in:
            out_samples = n_samples
        else:
            out_samples = int(n_samples * sr_out / sr_in)
        
        if y.shape[-1] < out_samples:
            y = torch.nn.functional.pad(y, (0, out_samples-y.shape[-1]))
        else:
            y = y[..., :out_samples]
 
        return y


    
if __name__ == "__main__":
    
    model = GAP_URGENet()

    x = torch.randn(2, 16000*4)
    
    y = model(x, sr_in=16000, sr_out=24000, enable_plc=True)
    print(y.shape)
    
    y = model(x, sr_in=16000, sr_out=44100, enable_plc=True)
    print(y.shape)
    
    y = model(x, sr_in=48000, sr_out=48000, enable_plc=True)
    print(y.shape)

    # from ptflops import get_model_complexity_info
    
    # with torch.inference_mode():
    #     macs, params = get_model_complexity_info(model, (16000,), print_per_layer_stat=False)
    
    # params = 0
    # for p in model.parameters():
    #     params += p.numel()
    # print(macs, f"{params / 1e6:.2f} M")

