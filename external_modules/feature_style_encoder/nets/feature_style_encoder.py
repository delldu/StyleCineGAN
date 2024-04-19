# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F

# from torch.nn.utils import spectral_norm
# from torchvision import models #, utils

from arcface.iresnet import *
import pdb

class fs_encoder_v2(nn.Module):
    def __init__(self, n_styles=18, opts=None, resnet_layer=[4, 5, 6], stride=(2, 2)):
    
        super().__init__()  

        # n_styles = 18
        # opts = Namespace(config='lhq_k10', real_dataset_path='', 
        #     dataset_path='', label_path='', 
        #     stylegan_model_path='./pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt', 
        #     w_mean_path='./pretrained_models/stylegan2-pytorch/sg2-lhq-1024-mean.pt', 
        #     arcface_model_path='./pretrained_models/backbone.pth', 
        #     log_path='./pretrained_models/logs/lhq_k10', checkpoint='', idx_k=10)
        
        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        # input conv layer
        self.conv = nn.Sequential(*list(resnet50.children())[:3])
        
        # define layers
        self.block_1 = list(resnet50.children())[3] # 15-18
        self.block_2 = list(resnet50.children())[4] # 10-14
        self.block_3 = list(resnet50.children())[5] # 5-9
        self.block_4 = list(resnet50.children())[6] # 1-4
        

        self.avg_pool = nn.AdaptiveAvgPool2d((3,3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

        self.idx_k = int(opts.idx_k)
        # self.feat_size = 128
        self.feat_ch = 256
        self.in_feat = 64
        
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(self.in_feat, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(self.in_feat, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            
            nn.Conv2d(512, self.feat_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.feat_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        )

    def forward(self, x):
        latents = []
        features = []
        x = self.conv(x) # size = (batch_size, 64, 256, 256)
        if self.idx_k in [10,11,12,13,14,15]: # True
            content = self.content_layer(x) # size = (batch_size, 256, 128, 128)
        x = self.block_1(x)  # torch.Size([batch_size, 64, 128, 128])
        
        features.append(self.avg_pool(x))
        x = self.block_2(x)

        features.append(self.avg_pool(x))
        x = self.block_3(x)
 
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        return out, content
