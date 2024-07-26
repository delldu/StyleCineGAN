import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import todos
import pdb

def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder_v2()
        self.scale = 2
        self.register_buffer('mean', torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.load_weights()

    def normalize(self, img):
        return (img - self.mean) / self.std

    def load_weights(self):
        cdir = os.path.dirname(__file__)
        model_path = "models/encoder_v2.pth" # -- 427 M
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.enc.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

        # load latent average ...
        w_mean_path = "models/sg2-lhq-1024-mean.pth"
        checkpoint = model_path if cdir == "" else cdir + "/" + w_mean_path
        print(f"Loading {checkpoint} ...")
        self.dlatent_avg = torch.load(checkpoint)


    def forward(self, img) -> List[torch.Tensor]: 
        img = self.normalize(img)

        # Reconstruction
        w_recon, fea = self.enc(downscale(img, self.scale, 'bilinear')) 
        w_recon = w_recon + self.dlatent_avg.to(img.device)

        return w_recon, fea


class Encoder_v2(nn.Module):
    def __init__(self, n_styles=18, stride=(2, 2)):
    
        super().__init__()  
        # n_styles = 18
        resnet50 = IResNet()
        # resnet50.load_state_dict(torch.load(opts.arcface_model_path)) # './pretrained_models/backbone.pth' -- 167M

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

        self.idx_k = 10 # int(opts.idx_k) # === 10
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
        # todos.debug.output_var("x", x)
        # tensor [x] size: [1, 3, 256, 256], min: -0.92598, max: 1.0, mean: -0.032175
    
        latents = []
        features = []
        x = self.conv(x) # size = (batch_size, 64, 256, 256)

        if self.idx_k in [10,11,12,13,14,15]: # True
            content = self.content_layer(x) # size = (batch_size, 256, 128, 128)
        else:
            pdb.set_trace()

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
        # todos.debug.output_var("latents", latents)
        # todos.debug.output_var("out", out)
        # print("-" * 80)
        # latents is list: len = 18
        #     tensor [item] size: [1, 512], min: -1.065482, max: 1.260337, mean: 0.064193
        #     tensor [item] size: [1, 512], min: -0.818316, max: 0.725723, mean: 0.020855
        #     tensor [item] size: [1, 512], min: -1.087323, max: 0.915459, mean: 0.030093
        #     tensor [item] size: [1, 512], min: -1.151149, max: 1.146161, mean: -0.059956
        #     tensor [item] size: [1, 512], min: -1.225104, max: 1.150664, mean: 0.001801
        #     tensor [item] size: [1, 512], min: -1.215691, max: 1.313415, mean: -0.056188
        #     tensor [item] size: [1, 512], min: -1.760906, max: 1.381754, mean: -0.052196
        #     tensor [item] size: [1, 512], min: -1.864859, max: 1.802828, mean: -0.169308
        #     tensor [item] size: [1, 512], min: -1.39873, max: 1.214812, mean: -0.062208
        #     tensor [item] size: [1, 512], min: -1.824914, max: 2.039525, mean: -0.081855
        #     tensor [item] size: [1, 512], min: -8.943651, max: 7.46369, mean: -0.052406
        #     tensor [item] size: [1, 512], min: -15.729791, max: 10.9655, mean: -0.220173
        #     tensor [item] size: [1, 512], min: -7.18438, max: 5.650012, mean: -0.163154
        #     tensor [item] size: [1, 512], min: -6.406906, max: 7.201956, mean: -0.074408
        #     tensor [item] size: [1, 512], min: -5.296788, max: 6.32128, mean: 0.047079
        #     tensor [item] size: [1, 512], min: -5.171273, max: 5.483113, mean: -0.040113
        #     tensor [item] size: [1, 512], min: -7.715606, max: 6.512413, mean: -0.049734
        #     tensor [item] size: [1, 512], min: -2.106955, max: 1.803118, mean: -0.012158
        # tensor [out] size: [1, 18, 512], min: -15.729791, max: 10.9655, mean: -0.051657
        # --------------------------------------------------------------------------------
        
        return out, content



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride
        # pdb.set_trace()

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    '''[3, 4, 14, 3] --- iresnet50'''
    fc_scale = 7 * 7
    def __init__(self,
            block=IBasicBlock, 
            layers=[3, 4, 14, 3],
            dropout=0, 
            num_features=512, 
            zero_init_residual=False,
            groups=1, 
            width_per_group=64, 
            # fp16=False
        ):
        super().__init__()
        # layers = [3, 4, 14, 3]
        # dropout = 0
        # num_features = 512
        # zero_init_residual = False
        # groups = 1
        # width_per_group = 64
        # fp16 = False


        # self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups # === 1
        self.base_width = width_per_group # === 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block,
                                       64, 
                                       layers[0], # 3
                                       stride=2,
                                    )
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1], # 4
                                       stride=2,
                                       dilate=False)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2], # 14
                                       stride=2,
                                       dilate=False)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3], # 3
                                       stride=2,
                                       dilate=False)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)

        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, 0, 0.1)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # if zero_init_residual: # False
        #     for m in self.modules():
        #         if isinstance(m, IBasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)
        # pdb.set_trace()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            pdb.set_trace()
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        # x = self.fc(x.float() if self.fp16 else x)

        x = self.features(x)

        return x
