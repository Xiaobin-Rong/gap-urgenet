# Copyright (c) 2025 Xiaobin-Rong.
# Adapted under MIT LICENSE.
# Source: https://github.com/Xiaobin-Rong/SEtrain

import os
import torch
import shutil
import random
import argparse
import numpy as np
import torch.distributed as dist
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pathlib import Path
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from utils.distributed_utils import reduce_value

from models.predictor.tfgridnet import TFGridNet as Model
from loaders.dataloader import URGENT3Dataset as Dataset
from utils.loss_factory import HybridLoss as Loss
from utils.scheduler import LinearWarmupCosineAnnealingLR as WarmupLR

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic =True


def run(rank, config, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(f"cuda:{rank}")
    
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    # config['train_dataloader']['batch_size'] = config['train_dataloader']['batch_size'] // args.world_size
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
    
    model = Model(**config['predictor_config']).to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['optimizer']['lr'], betas=(0.8, 0.99), weight_decay=0.01)
    scheduler = WarmupLR(optimizer, **config['scheduler'])
    
    loss = Loss().to(args.device)

    trainer = Trainer(
        config=config, model=model, optimizer=optimizer, scheduler=scheduler, loss_func=loss,
        train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
        train_sampler=train_sampler, args=args
    )

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size
        
        self.default_fs = config['samplerate']

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'code')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        # save the config
        if self.rank == 0:
            shutil.copy2(__file__, self.exp_path)
            shutil.copy2(args.config, Path(self.exp_path) / 'config.yaml')
            
            for file in Path(__file__).parent.parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            for d in ['configs', 'loaders', 'models', 'utils']:
                shutil.copytree(Path(__file__).parent.parent / d, Path(self.code_path) / d, dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 1e8

        if self.resume:
            self._resume_checkpoint()

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))

        if score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _del_checkpoint(self, epoch, score):
        # Condition 1: epoch-1 is not a multiple of self.save_checkpoint_interval;
        # Condition 2: current score is best score.
        # if (epoch - 1) % self.save_checkpoint_interval != 0 or score == self.best_score:
        if (epoch - 1) % self.save_checkpoint_interval != 0:
            prev_epoch = epoch - 1
            checkpoint_file = os.path.join(self.checkpoint_path, f'model_{str(prev_epoch).zfill(3)}.tar')
        
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    print(f"Deleted checkpoint: {checkpoint_file}")
                except Exception as e:
                    print(f"Failed to delete checkpoint {checkpoint_file}: {e}")

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.scheduler.warmup_steps = 5000
        # self.scheduler.decay_until_step = 50000
        # if self.rank == 0:
        #     print(self.scheduler.state_dict())
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _train_epoch(self, epoch):
        total_loss = 0
        self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=120)

        for step, (noisy, clean, info) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            fs = int(info['fs'][0])

            enhanced = self.model(noisy, fs)

            loss = self.loss_func(enhanced, clean, fs)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            self.train_bar.desc = '   train[{}/{}][{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, fs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'train_loss={:.3f}'.format(total_loss / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()
            self.scheduler.step()

        # 等待所有进程计算完毕
        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0

        self.validation_bar = tqdm(self.validation_dataloader, ncols=120)
        for step, (noisy, clean, info) in enumerate(self.validation_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            fs = int(info['fs'][0])

            enhanced = self.model(noisy, fs)  # (B,F,T,2)
 
            loss = self.loss_func(enhanced, clean, fs)
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            if self.rank == 0 and (epoch<10 or epoch%10 == 0) and step <= 10:
            # if self.rank == 0 and step <= 3:
                noisy_path = os.path.join(self.sample_path, '{}_noisy.wav'.format(info['id'][0]))
                clean_path = os.path.join(self.sample_path, '{}_clean.wav'.format(info['id'][0]))
                enhanced_path = os.path.join(self.sample_path, '{}_enh_epoch{}.wav'.format(info['id'][0], str(epoch).zfill(3)))
                scale = 0.9 / (max([noisy.abs().max(), clean.abs().max(), enhanced.abs().max()]).item() + 1e-12)
            
                sf.write(noisy_path, noisy.cpu().numpy().squeeze() * scale, fs)
                sf.write(clean_path, clean.cpu().numpy().squeeze() * scale, fs)
                sf.write(enhanced_path, enhanced.cpu().numpy().squeeze() * scale, fs)

            self.validation_bar.desc = 'validate[{}/{}][{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, fs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'valid_loss={:.3f}'.format(total_loss / step)

        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('val_loss', {'val_loss': total_loss / step}, epoch)

        return total_loss / step

    def train(self):
        if self.resume:
            self._resume_checkpoint()

        if self.start_epoch == 1:
            self._set_eval_mode()
            _ = self._validation_epoch(0)

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss = self._validation_epoch(epoch)
            torch.cuda.empty_cache()

            # if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
            if self.rank == 0:
                self._save_checkpoint(epoch, valid_loss)
                self._del_checkpoint(epoch, valid_loss)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

            print('------------Training for {} epochs has done!------------'.format(self.epochs))


if __name__ == '__main__':
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train_predictor.yaml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = OmegaConf.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
