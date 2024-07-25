import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
device_ids = [0]
device = torch.device('cuda')
import pdb

def load_stylegan2(ckpt_dir, channel_multiplier=2):
    '''
    <Input>
    ckpt_dir:      checkpoint direcotry    (str)    
    
    <Output>
    g_ema:         stylegan2 model
    '''
    
    print("\n>>> Loading StyleGAN...")
    
    from models.stylegan2.model import Generator
    g_ema = Generator(1024, 512, 8, channel_multiplier=channel_multiplier)
    g_ckpt = torch.load(ckpt_dir)
    # ckpt_dir ---- './pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt'

    g_ema.load_state_dict(g_ckpt["g_ema"])
    g_ema = g_ema.eval().cuda()
    g_ema = nn.DataParallel(g_ema, device_ids=device_ids)
    
    print(">>> Loading done ------------------ \n")

    return g_ema # g_ema.size() -- 127M


def load_encoder(ckpt_dir, recon_idx=10):

    from argparse import Namespace

    print(">>> Loading StyleGAN encoder...")
    
    import os
    encoder_opts, encoder_config = set_encoder_args(os.getcwd(), ckpt_dir, recon_idx)
    trainer = load_fs_encoder(encoder_opts, encoder_config)
    print(">>> Loading done ------------------ \n")
    return trainer
 

# -----------------------------------------------------------------------------------------
def set_encoder_args(base_dir, pretrained_dir, idx_k):
    
    fs_dir = f"{base_dir}/external_modules/feature_style_encoder"
    
    opts = {
        'config': '',
        'real_dataset_path': '',
        'dataset_path': '',
        'label_path': '',
        'stylegan_model_path': '',
        'w_mean_path': '',
        'arcface_model_path': '',
        'log_path': '',
        'checkpoint': ''
    }
    
    opts['config'] = f"lhq_k{idx_k}"
    opts['log_path'] = './pretrained_models/logs/lhq_k10' # f"{pretrained_dir}/logs/{opts['config']}"
    opts['stylegan_model_path'] = f"{pretrained_dir}/stylegan2-pytorch/sg2-lhq-1024.pt"
    opts['w_mean_path'] = f"{pretrained_dir}/stylegan2-pytorch/sg2-lhq-1024-mean.pt"
    opts['arcface_model_path'] = f"{pretrained_dir}/backbone.pth"
    
    from argparse import Namespace
    opts = Namespace(**opts)
    
    import yaml
    config = yaml.load(open(f'{fs_dir}/configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
    opts.idx_k = 10 # config['idx_k']
    
    # Namespace(config='lhq_k10', real_dataset_path='', dataset_path='', label_path='', 
    #     stylegan_model_path='./pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt', 
    #     w_mean_path='./pretrained_models/stylegan2-pytorch/sg2-lhq-1024-mean.pt', 
    #     arcface_model_path='./pretrained_models/backbone.pth', 
    #     log_path='./pretrained_models/logs/lhq_k10', checkpoint='', idx_k=10)

    return opts, config


def load_fs_encoder(opts, config):
    
    import sys
    sys.path.append("./external_modules/feature_style_encoder")
    from trainerx import Trainer # rename train.py ==> trainx.py
    
    trainer = Trainer(config, opts)
    trainer.initialize(opts.w_mean_path)   
    trainer.load_model(opts.log_path)
    trainer.enc.eval() # fs_encoder_v2(...)

    trainer.to(device)

    # (Pdb) opts.log_path -- './pretrained_models/logs/lhq_k10'

    return trainer

        
