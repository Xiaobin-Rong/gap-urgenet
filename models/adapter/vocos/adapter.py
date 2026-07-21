import torch
from torch import nn
from omegaconf import OmegaConf
from .backbone import VocosBackbone as Decoder


class VocosAdapter(nn.Module):
    def __init__(
        self,
        input_channels=1024,
        dim=1024,
        intermediate_dim=4096,
        num_layers=12,
        output_channels=1024
    ):
        super().__init__()
        self.proj = nn.Linear(input_channels, input_channels)
        self.decoder = Decoder(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers
        )
        self.head = nn.Linear(dim, output_channels)
        
    @classmethod
    def from_pretrained(cls, model_ckpt, frozen=True):
        print("[Adapter] Loading from", model_ckpt)
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
        
    
    def forward(
        self,
        embed_a,
        embed_p
    ):
        """embed_p: (B, T, D), embed_a: (B, T, D)"""
        embed_p = self.proj(embed_p)
        
        x = embed_p + embed_a
        
        x = x.transpose(1,2)
        x = self.decoder(x)
        x = x.transpose(1,2)
        
        x = self.head(x)

        return x  # (B, T, D)
    

if __name__ == "__main__":
    
    config = OmegaConf.load('configs/cfg_train_adapter.yaml')

    adapter = VocosAdapter(**config['adapter_config'])
    
    params = sum([item.numel() for item in adapter.parameters()])
    print(f"params: {params/1e6:.2f} M")
    
    # x_p = torch.randn(1, 1024, 50)
    # x_a = torch.randn(1, 1024, 50)
    # out = adapter(x_p, x_a)
    # print(out.shape)

    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(wavlmdec, (1024, 50), as_strings=True,
    #                                         print_per_layer_stat=False, verbose=False)
    # print(f"MACs: {macs}, Params: {params}")