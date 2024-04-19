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
        self.scale = int(np.log2(config['resolution']/config['enc_resolution']))
        self.scale_mode = 'bilinear'
        self.opts = opts
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2
        self.idx_k = 5
        if 'idx_k' in self.config:
            self.idx_k = self.config['idx_k']
        if 'stylegan_version' in self.config and self.config['stylegan_version'] == 3:
            self.n_styles = 16

        # Networks
        # in_channels = 256
        # if 'in_c' in self.config: # False
        #     in_channels = config['in_c']
        enc_residual = False
        if 'enc_residual' in self.config: # True
            enc_residual = self.config['enc_residual'] # False

        enc_residual_coeff = False
        if 'enc_residual_coeff' in self.config: # False
            enc_residual_coeff = self.config['enc_residual_coeff']

        resnet_layers = [4,5,6]
        if 'enc_start_layer' in self.config: # False
            st_l = self.config['enc_start_layer']
            resnet_layers = [st_l, st_l+1, st_l+2]

        if 'scale_mode' in self.config: # False
            self.scale_mode = self.config['scale_mode']

        # Load encoder
        self.stride = (self.config['fs_stride'], self.config['fs_stride'])
        self.enc = fs_encoder_v2(n_styles=self.n_styles, # 18
                                 opts=opts, 
                                 residual=enc_residual, # False
                                 use_coeff=enc_residual_coeff, 
                                 resnet_layer=resnet_layers, 
                                 stride=self.stride, # (2, 2)
                                ) # Model Size() -- 427
        

        ##########################
        self.StyleGAN = P2S2PGenerator(1024, 512, 8)

    def initialize(self, stylegan_model_path, arcface_model_path, parsing_model_path, w_mean_path):
        # load StyleGAN model
        # stylegan_model_path = './pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt'
        # arcface_model_path = './pretrained_models/backbone.pth'
        # parsing_model_path = './pretrained_models/79999_iter.pth'
        # w_mean_path = './pretrained_models/stylegan2-pytorch/sg2-lhq-1024-mean.pt'

        try:
            stylegan_state_dict = torch.load(stylegan_model_path, map_location='cpu')
            self.StyleGAN.load_state_dict(get_keys(stylegan_state_dict, 'decoder'), strict=True)
            self.StyleGAN.to(self.device)
            self.dlatent_avg = stylegan_state_dict['latent_avg'].to(self.device)
        except:
            # ===> pdb.set_trace() for g_ema !!!
            state_dict = torch.load(stylegan_model_path)
            self.StyleGAN.load_state_dict(state_dict["g_ema"])
            self.StyleGAN.to(self.device)
            with torch.no_grad():
                self.dlatent_avg = torch.load(w_mean_path).to(self.device)


    def get_image(self, w=None, img=None, noise=None, zero_noise_input=True, training_mode=True):
        # w = None
        # noise = None
        # zero_noise_input = True
        # training_mode = False

        x_1, n_1 = img, noise
        w_delta = None
        fea = None
        features = None
        # Reconstruction
        k = self.idx_k
        w_recon, fea = self.enc(downscale(x_1, self.scale, self.scale_mode)) 
        w_recon = w_recon + self.dlatent_avg
        features = [None]*k + [fea] + [None]*(17-k)

 
        # generate image
        x_1_recon, fea_recon = self.StyleGAN([w_recon], input_is_latent=True, 
            return_features=True, features_in=features, feature_scale=min(1.0, 10.0))
        
        fea_recon = fea_recon[k].detach()


        return [x_1_recon, x_1[:,:3,:,:], w_recon, w_delta, n_1, fea, fea_recon]

    def test(self, w=None, img=None, noise=None, zero_noise_input=True, return_latent=True, training_mode=False):        
        if 'n_iter' not in self.__dict__.keys():
            self.n_iter = 1e5
        out = self.get_image(w=w, img=img, noise=noise, training_mode=training_mode)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
        output = [x_1, x_1_recon]
        if return_latent: # True
            output += [w_recon, fea_1]
        return output

    def load_model(self, log_dir):
        self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))
