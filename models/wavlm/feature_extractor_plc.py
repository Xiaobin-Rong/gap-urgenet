# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# License included under licenses/LICENSE_pase.

"""
WavLM feature extractor tailored for primary PLC.  
The process involves:  
1. Detecting packet loss regions;  
2. Generating the corresponding mask indices;  
3. Applying masks to the CNN output at the specified regions;
4. Performing the standard WavLM forward pass.  
"""

import torch
import torch.nn as nn
from typing import Union, List
from .WavLM import WavLMConfig, WavLM
from torchaudio.functional import resample


class WavLM_feat(nn.Module):
    def __init__(
        self,
        wavlm_ckpt_path="/data/hdd0/Pre-trained/WavLM/WavLM-Large.pt",
        output_layer: Union[int, List[int]] = 24,
        load_pretrained: bool=True,
        frozen: bool=True
    ):
        super().__init__()
        self.wavlm_ckpt_path = wavlm_ckpt_path
        cpt = torch.load(self.wavlm_ckpt_path, map_location="cpu")
        print("[WavLM] Output layer:", output_layer)
        
        self.cfg = WavLMConfig(cpt['cfg'])
        self.wavlm = WavLM(self.cfg)
        if load_pretrained:
            self.wavlm.load_state_dict(cpt['model'])
            print("[WavLM] Loading from:", wavlm_ckpt_path)
        
        if frozen:
            self.wavlm.eval()
            for p in self.wavlm.parameters():
                p.requires_grad = False

        self.output_layer = output_layer

    @staticmethod
    def pad(x):
        if x.shape[1] % 320 != 80:
            pad_points = x.shape[1]//320 * 320 + 80 - x.shape[1]
            x = nn.functional.pad(x, [0, pad_points])
        return x


    def forward(
        self, 
        wav,
        sr=None,
        mask=True, 
        mask_indices=None,
        return_mask_indices=False
    ):
        """wav: (B, 1, L)"""
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        
        if (sr is not None) and (sr != 16000):
            wav = resample(wav, orig_freq=sr, new_freq=16000)
            
        wav = self.pad(wav)
        
        L = self.output_layer if isinstance(self.output_layer, int) else max(self.output_layer)
        
        if mask:
            if mask_indices is None:
                mask_indices = detect_packet_loss_indices(wav, fs=16000)
        
        res = self.wavlm.extract_features(wav, output_layer=L, mask=mask, mask_indices=mask_indices)[0]
        layer_reps = res["layer_reps"]
        
        if isinstance(self.output_layer, int) or len(self.output_layer) == 1:
            feat = layer_reps[L]
            feat = torch.nn.functional.layer_norm(feat, feat.shape[1:], eps=1e-6)
        else:
            feat = []
            for i in range(len(self.output_layer)):
                feat_i = layer_reps[self.output_layer[i]]
                feat_i = torch.nn.functional.layer_norm(feat_i, feat_i.shape[1:], eps=1e-6)
                feat.append(feat_i)

        if not return_mask_indices:
            return feat
        else:
            return feat, mask_indices

    

def detect_packet_loss_indices(
    audio_batch: torch.Tensor,
    fs: int,
    packet_duration_ms: int = 20,
    threshold: float = 1e-7,
    min_zero_ratio: float = 0.99,
):
    """
    Detect packet loss regions for each audio in a batch in parallel, 
    and return the indices of lost packets for each audio.

    Args:
        audio_batch (tensor): shape (B, L)
        fs (int): sampling rate
        packet_duration_ms (int): packet duration in milliseconds
        threshold (float): threshold to consider a value as zero
        min_zero_ratio (float): minimum ratio of near-zero samples to consider a packet lost

    Returns:
        Tensor: a mask indicating lost packets for each audio
    """
    B, L = audio_batch.shape
    packet_len = int(fs * packet_duration_ms / 1000)
    num_packets = L // packet_len
    if num_packets == 0:
        return None
    # Truncate extra samples and reshape to (B, num_packets, packet_len)
    audio_batch = audio_batch[:, :num_packets * packet_len]
    packets = audio_batch.view(B, num_packets, packet_len)
    # Compute the ratio of samples below the threshold for each packet (B, num_packets)
    zero_ratio = (packets.abs() < threshold).float().mean(dim=2)
    # Obtain mask indicating lost packets
    loss_mask = zero_ratio >= min_zero_ratio  # (B, num_packets)

    # # Obtain indices of lost packets for each audio
    # loss_indices = [
    #     (torch.nonzero(loss_mask[b], as_tuple=False).squeeze(-1))  # return indices of lost packets
    #     for b in range(B)
    # ]
    # torch.stack(loss_indices, dim=0)
    
    return loss_mask


if __name__ == "__main__":
    model = WavLM_feat(output_layer=[1, 24])
    
    params = sum([item.numel() for item in model.parameters()])
    print(f"params: {params/1e6:.2f} M")
    
    # print(dir(feature_extractor.cfg)) 
    
    # for k in dir(feature_extractor.cfg):
    #     # if 'mask' in k:
    #     print(k, getattr(feature_extractor.cfg, k))
    # x = torch.randn(1, 16000)
    # y = feature_extractor(x)
    # print(y[0].shape)
    # print(y[-1].shape)
