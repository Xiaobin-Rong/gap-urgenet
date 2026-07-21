# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# License included under licenses/LICENSE_pase.

import torch
from torch import nn
from omegaconf import OmegaConf
from .backbone import VocosBackbone as Decoder
from .head import ISTFTHead as Head


class VocosVocoder(nn.Module):
    def __init__(
        self,
        input_channels=1024,
        dim=768,
        intermediate_dim=2304,
        num_layers=12,
        num_res: int = 4,
        num_attn: int = 1,
        n_fft=1280,
        hop_length=320
    ):
        super().__init__()
        self.decoder = Decoder(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_res=num_res,
            num_attn=num_attn
        )
        self.head = Head(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    @classmethod
    def from_pretrained(cls, model_ckpt, frozen=True):
        print("[Vocoder] Loading from", model_ckpt)
        ckpt = torch.load(model_ckpt, map_location='cpu')
        cfg = ckpt['cfg']

        model = cls(**cfg)
        
        if 'generator' in ckpt.keys():
            model_dict = ckpt['generator']
        elif 'model' in ckpt.keys():
            model_dict = ckpt['model']
            
        model.load_state_dict(model_dict, strict=False)
        
        if frozen:
            model = model.eval()
        
            for p in model.parameters():
                p.requires_grad = False
        
        return model


    def forward(self, input):
        """embed: (B, D, T)"""
        x = self.decoder(input)
        audio_output = self.head(x)
    
        return audio_output
    
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train_vocoder.yaml')

    wavlmdec = VocosVocoder(**config['decoder_config'])
    
    # x = torch.randn(1, 1024, 50)
    # out = wavlmdec(x)
    # print(out.shape)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(wavlmdec, (1024, 50), as_strings=True,
                                            print_per_layer_stat=False, verbose=False)
    print(f"MACs: {macs}, Params: {params}")