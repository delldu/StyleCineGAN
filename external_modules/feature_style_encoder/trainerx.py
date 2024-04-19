import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('pixel2style2pixel/')
from pixel2style2pixel.models.stylegan2.model import P2S2PGenerator, get_keys


from nets.feature_style_encoder import fs_encoder_v2
import pdb

def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x

class Trainer(nn.Module):
    def __init__(self, config, opts):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.config = config
        self.device = torch.device(self.config['device'])
        self.scale = int(np.log2(config['resolution']/config['enc_resolution'])) # 2
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2
        self.idx_k = 10

        # Networks
        enc_residual = False
        enc_residual_coeff = False
        resnet_layers = [4,5,6]

        # Load encoder
        self.stride = (self.config['fs_stride'], self.config['fs_stride'])
        self.enc = fs_encoder_v2(n_styles=self.n_styles, # 18
                                 opts=opts, 
                                 residual=enc_residual, # False
                                 use_coeff=enc_residual_coeff, 
                                 resnet_layer=resnet_layers, 
                                 stride=self.stride, # (2, 2)
                                ) # Model Size() -- 427 M
        

        ##########################
        self.StyleGAN = P2S2PGenerator(1024, 512, 8)

    def initialize(self, stylegan_model_path, arcface_model_path, parsing_model_path, w_mean_path):
        # load StyleGAN model
        # stylegan_model_path = './pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt'
        # arcface_model_path = './pretrained_models/backbone.pth'
        # parsing_model_path = './pretrained_models/79999_iter.pth'
        # w_mean_path = './pretrained_models/stylegan2-pytorch/sg2-lhq-1024-mean.pt'

        state_dict = torch.load(stylegan_model_path)
        self.StyleGAN.load_state_dict(state_dict["g_ema"])
        self.StyleGAN.to(self.device)
        with torch.no_grad():
            self.dlatent_avg = torch.load(w_mean_path).to(self.device)

    def get_image(self, img=None):
        x_1 = img
        # Reconstruction
        k = self.idx_k # === 10
        w_recon, fea = self.enc(downscale(x_1, self.scale, 'bilinear')) 
        w_recon = w_recon + self.dlatent_avg
        features = [None]*k + [fea] + [None]*(17-k)

        # generate image
        x_1_recon, fea_recon = self.StyleGAN([w_recon], features_in=features)
        fea_recon = fea_recon[k].detach()
        return [x_1_recon, x_1[:,:3,:,:], w_recon, fea, fea_recon]

    def test(self, img=None):        
        out = self.get_image(img=img)
        x_1_recon, x_1, w_recon, fea_1 = out[:4]
        output = [x_1, x_1_recon, w_recon, fea_1]
        return output

    def load_model(self, log_dir):
        self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))
