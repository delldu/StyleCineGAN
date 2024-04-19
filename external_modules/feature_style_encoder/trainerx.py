import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('pixel2style2pixel/')
# from pixel2style2pixel.models.stylegan2.model import P2S2PGenerator, get_keys


from nets.feature_style_encoder import fs_encoder_v2
import pdb

def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x

class Trainer(nn.Module):
    def __init__(self, config, opts):
        super().__init__()
        # Load Hyperparameters
        self.config = config
        self.device = torch.device(self.config['device'])
        self.scale = int(np.log2(config['resolution']/config['enc_resolution'])) # 2
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2

        # Networks
        # enc_residual = False
        # enc_residual_coeff = False
        # resnet_layers = [4,5,6]

        # Load encoder
        self.stride = (self.config['fs_stride'], self.config['fs_stride'])
        self.enc = fs_encoder_v2(n_styles=self.n_styles, # 18
                                 opts=opts, 
                                 # residual=enc_residual, # False
                                 # use_coeff=enc_residual_coeff, 
                                 # resnet_layer=resnet_layers, 
                                 stride=self.stride, # (2, 2)
                                ) # Model Size() -- 427 M
        


    def initialize(self, w_mean_path):
        with torch.no_grad():
            self.dlatent_avg = torch.load(w_mean_path).to(self.device)

    def get_image(self, img=None):
        x_1 = img
        # Reconstruction
        w_recon, fea = self.enc(downscale(x_1, self.scale, 'bilinear')) 
        w_recon = w_recon + self.dlatent_avg
        return [w_recon, fea]

    def test(self, img=None):        
        out = self.get_image(img=img)
        w_recon, fea = out[:2]
        output = [w_recon, fea]
        return output

    def load_model(self, log_dir):
        # pretrained_models/logs/lhq_k10/enc.pth.tar ????
        self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))
