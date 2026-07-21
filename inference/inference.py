# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Adapted under Apache-2.0
# Source: https://github.com/cisco-open/pase/blob/main/inference/inference.py
# License included under licenses/LICENSE_pase.

import torch
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from models.gap_urgenet import GAP_URGENet


REPO_ID = "Xiaobin-Rong/gap-urgenet"

def get_checkpoint_path(ckpt_arg, filename, download_dir=None):
    """
    Ensures the availability of a checkpoint file by resolving local paths 
    or downloading from the HuggingFace Hub.

    Args:
        ckpt_arg (str): Local path override. If it exists, use this path.
        filename (str): Filename to retrieve from the remote repository.
        download_dir (str, optional): Custom download destination. 

    Returns:
        str: Absolute path to the checkpoint.
    """
    
    if ckpt_arg and os.path.exists(ckpt_arg):
        full_path = os.path.abspath(ckpt_arg)
        print(f"[*] Using user-specified local checkpoint: {full_path}")
        return full_path
    
    print(f"[*] Downloading {filename} from Hugging Face ({REPO_ID})...")
    
    if not os.path.exists(f"{download_dir}/config.json"):
        hf_hub_download(
            repo_id=REPO_ID,
            filename="config.json",
            local_dir=download_dir,
            local_dir_use_symlinks=False,
        )
        
    path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=filename, 
        local_dir=download_dir,
        local_dir_use_symlinks=False
    )

    absolute_path = os.path.abspath(path)
    print(f"[*] Checkpoint is stored at: {absolute_path}")
    
    return absolute_path


def inference_file(input_file, output_file, model, **kwargs):
    """
    Run inference on a single audio file and save the result.
    Args:
        input_file (str): Path to input audio file.
        output_file (str): Path to save enhanced audio.
        model: Initialized model for inference.
        **kwargs: Additional keyword arguments.
    """
    audio, fs = sf.read(input_file, dtype='float32')
    input_tensor = torch.FloatTensor(audio).unsqueeze(
        0).to(next(model.parameters()).device)
    
    sr_out = kwargs.get('sr_out', None)
    enable_plc = kwargs.get('enable_plc', True)
    
    if sr_out is None:
        sr_out = fs
        
    with torch.inference_mode():
        output = model(input_tensor, sr_in=fs, 
                       sr_out=sr_out, enable_plc=enable_plc)
    enhanced = output.cpu().detach().numpy().squeeze()
    
    scale = np.max(np.abs(audio))
    enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8) * scale
    
    sf.write(output_file, enhanced, sr_out)


def inference_folder(input_dir, output_dir, model, extension='.wav', **kwargs):
    """
    Run inference on all audio files in a folder and save results to output_dir.
    Args:
        input_dir (str): Directory with input audio files.
        output_dir (str): Directory to save enhanced files.
        model: Initialized model for inference.
        extension (str): File extension to filter (default: '.wav').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(extension, str):
        extensions = [ext.strip().lower() for ext in extension.split(',')]
    else:
        extensions = extension
    
    for fname in tqdm(os.listdir(input_dir)):
        if fname.lower().endswith(tuple(extensions)):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            inference_file(in_path, out_path, model, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run GAP-URGENet inference on audio files.")
    parser.add_argument('-I', '--input_dir', type=str, required=True,
                        help='Input directory with audio files')
    parser.add_argument('-O', '--output_dir', type=str, required=True,
                        help='Output directory for enhanced files')
    parser.add_argument('-D', '--device', type=str, default='cuda:0',
                        help='Torch device (default: cuda:0)')
    parser.add_argument('-E', '--extension', type=str, default='.wav',
                        help='Audio file extensions separated by commas (e.g., .wav,.flac)')
    parser.add_argument('--sr_out', type=int, default=None,
                        help='Output sampling rate (default: same as input)')
    parser.add_argument('--enable_plc', type=bool, default=True,
                        help='Whether to perform packet loss concealment (PLC)')
    
    parser.add_argument('--dewavlm_ckpt', type=str, default=None, 
                        help='Path to DeWavLM-Omni.pt (if None, download from HF)')
    parser.add_argument('--adapter_ckpt', type=str, default=None, 
                        help='Path to Adapter.pt (if None, download from HF)')
    parser.add_argument('--vocoder_ckpt', type=str, default=None, 
                        help='Path to Vocoder_DWO-L1.pt (if None, download from HF)')
    parser.add_argument('--predictor_ckpt', type=str, default=None, 
                        help='Path to Predictor.pt (if None, download from HF)')
    parser.add_argument('--postnet_ckpt', type=str, default=None, 
                        help='Path to PostNet.pt (if None, download from HF)')
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Directory to download checkpoints (if None, use HF default cache directory)')
    
    args = parser.parse_args()
    
    resolved_dewavlm_path = get_checkpoint_path(
        args.dewavlm_ckpt, "DeWavLM-Omni.pt", args.download_dir
    )
    resolved_adapter_path = get_checkpoint_path(
        args.adapter_ckpt, "Adapter.pt", args.download_dir
    )
    resolved_vocoder_path = get_checkpoint_path(
        args.vocoder_ckpt, "Vocoder.pt", args.download_dir
    )
    resolved_predictor_path = get_checkpoint_path(
        args.predictor_ckpt, "Predictor.pt", args.download_dir
    )
    resolved_postnet_path = get_checkpoint_path(
        args.postnet_ckpt, "PostNet.pt", args.download_dir
    )

    device = torch.device(args.device)
    model = GAP_URGENet(
        dewavlm_ckpt_path=resolved_dewavlm_path,
        adapter_ckpt_path=resolved_adapter_path,
        vocoder_ckpt_path=resolved_vocoder_path,
        predictor_ckpt_path=resolved_predictor_path,
        postnet_ckpt_path=resolved_postnet_path
    ).to(device).eval()

    inference_folder(args.input_dir, args.output_dir,
                     model, extension=args.extension,
                     sr_out=args.sr_out, enable_plc=args.enable_plc)