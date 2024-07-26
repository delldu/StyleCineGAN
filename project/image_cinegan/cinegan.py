import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import numpy as np
import cupy
import re

import todos
import pdb

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        # kernel = tensor([[0.062500, 0.187500, 0.187500, 0.062500],
        #         [0.187500, 0.562500, 0.562500, 0.187500],
        #         [0.187500, 0.562500, 0.562500, 0.187500],
        #         [0.062500, 0.187500, 0.187500, 0.062500]])
        # factor = 2
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1) # === (2, 1) ?

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=2):
        super().__init__()

        kernel = make_kernel(kernel)
        kernel = kernel * (upsample_factor ** 2)
        # kernel.size() -- [4, 4]
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        # u1d1
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(1.0))
        self.scale = (1.0 / math.sqrt(in_dim))

    def forward(self, input):
        return F.linear(input, self.weight * self.scale, bias=self.bias)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class EqualLinearWithLReLu(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=0.01):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(0.0))
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        return fused_leaky_relu(out, self.bias * self.lr_mul)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'

class ModConv2d(nn.Module):
    '''Without Upsample'''
    def __init__(self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1.0 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.modulation = EqualLinear(style_dim, in_channel)


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})'

    def forward(self, input, style):
        B, C, H, W = input.shape

        style = self.modulation(style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style

        # Modulated, Modulated, Modulated !!!
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, self.out_channel, 1, 1, 1)
        weight = weight.view(B * self.out_channel, C, self.kernel_size, self.kernel_size)

        # without upsample
        input = input.view(1, B * C, H, W)
        out = F.conv2d(input, weight, padding=self.padding, groups=B)

        _, _, H, W = out.shape
        out = out.view(B, self.out_channel, H, W)

        return out

class ModConv2dWithUpsample(nn.Module):
    '''ModConv2d With Upsample 2'''
    def __init__(self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        factor = 2
        p = (len([1, 3, 3, 1]) - factor) - (kernel_size - 1) # ==> 3 - kernel_size
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1
        self.blur = Blur([1, 3, 3, 1], pad=(pad0, pad1), upsample_factor=factor) # factor === 2

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel)


    def forward(self, input, style):
        B, C, H, W = input.shape

        style = self.modulation(style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style

        # Modulated, Modulated, Modulated !!!
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, self.out_channel, 1, 1, 1)
        weight = weight.view(B * self.out_channel, C, self.kernel_size, self.kernel_size)
        weight = weight.view(B, self.out_channel, C, self.kernel_size, self.kernel_size)
        weight = weight.transpose(1, 2).reshape(B * C, self.out_channel, self.kernel_size, self.kernel_size)

        input = input.view(1, B * C, H, W)
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=B)

        _, _, H, W = out.shape
        out = out.view(B, self.out_channel, H, W)
        out = self.blur(out)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})'


class Conv2dRGB(nn.Module):
    def __init__(self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel)

    def forward(self, input, style):
        B, C, H, W = input.shape

        style = self.modulation(style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        weight = weight.view(B * self.out_channel, C, self.kernel_size, self.kernel_size)

        input = input.view(1, B * C, H, W)
        out = F.conv2d(input, weight, padding=self.padding, groups=B)

        _, _, H, W = out.shape
        out = out.view(B, self.out_channel, H, W)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})'


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        B = input.shape[0]
        return self.input.repeat(B, 1, 1, 1)


class StyledConv(nn.Module):
    def __init__(self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False, # True | False
    ):
        super().__init__()
        if upsample:
            # ModConv2d With Upsample
            self.conv = ModConv2dWithUpsample(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
            )
        else:
            # ModConv2d Without Upsample
            self.conv = ModConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
            )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out
    

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.conv = Conv2dRGB(in_channel, 3, 1, style_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        return out

class ToRGBUpsample(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.upsample = Upsample([1, 3, 3, 1])
        self.conv = Conv2dRGB(in_channel, 3, 1, style_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip):
        out = self.conv(input, style)
        out = out + self.bias
        skip = self.upsample(skip)
        out = out + skip

        return out

class ToRGBUpsampleSkipNone(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.upsample = Upsample([1, 3, 3, 1]) # useless !!!
        self.conv = Conv2dRGB(in_channel, 3, 1, style_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style):
        out = self.conv(input, style)
        out = out + self.bias

        return out


class Generator(nn.Module):
    def __init__(self,
            size=1024,
            style_dim=512,
            n_mlp=8,
            channel_multiplier=2,
            lr_mlp=0.01,
    ):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinearWithLReLu(style_dim, style_dim, lr_mul=lr_mlp))

        self.style = nn.Sequential(*layers)
        # (Pdb) self.style
        # Sequential(
        #   (0): PixelNorm()
        #   (1): EqualLinear(512, 512)
        #   (2): EqualLinear(512, 512)
        #   (3): EqualLinear(512, 512)
        #   (4): EqualLinear(512, 512)
        #   (5): EqualLinear(512, 512)
        #   (6): EqualLinear(512, 512)
        #   (7): EqualLinear(512, 512)
        #   (8): EqualLinear(512, 512)
        # )

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier, # channel_multiplier === 2
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        channel_512 = channels[4]

        self.input = ConstantInput(channel_512)
        self.conv1 = StyledConv(channel_512, channel_512, 3, style_dim)
        self.to_rgb1 = ToRGB(channel_512, style_dim)

        self.log_size = int(math.log(size, 2)) # 10
        self.num_layers = (self.log_size - 2) * 2 + 1 # ==> 17

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = channel_512

        for layer_idx in range(self.num_layers): # self.num_layers === 17
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1): # self.log_size === 10 ==> 3, 4, 5, 6, 7, 8, 9, 10
            out_channel = channels[2 ** i]

            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim))
            # i === 7 ==> ToRGBUpsampleSkipNone            
            if i == 7:
                self.to_rgbs.append(ToRGBUpsampleSkipNone(out_channel, style_dim))
            else:
                self.to_rgbs.append(ToRGBUpsample(out_channel, style_dim))

            in_channel = out_channel

        self.load_weights()

        # self.convs1 = ...
        # self.convs2 = ...
        # self.noise1 = ...
        # self.noise2 = ...
        # self.n_latent = self.log_size * 2 - 2 # self.n_latent === 18
        # pdb.set_trace()
        
    def load_weights(self, model_path="models/sg2-lhq-1024.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu'))["g_ema"]) # 127M

    def forward(self, latent, feature, idx, n_frames, flow):
        recon_feature_idx = 10

        noise = [
            getattr(self.noises, f'noise_{i}') for i in range(self.num_layers) # self.num_layers === 17
        ]
        # 4, 8, 8, 16, 16, 32, 32, 64, 64,  128, 128, 256, 256, 512, 512, 1024, 1024
        # self.noises.noise_0.size() -- [1, 1, 4, 4]
        # self.noises.noise_16.size() -- [1, 1, 1024, 1024]

        # latent.size() -- [1, 18, 512]
        # latent = styles[0]
            
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        # todos.debug.output_var("skip", skip)
        # tensor [skip] size: [1, 3, 4, 4], min: -0.005031, max: 0.001965, mean: -0.001057
        
        # len(self.convs) -- 16, len(self.to_rgbs) --- 8
        i = 1 # ===> i = 1, 3, 5, 7, 9, 11, 13, 15
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], 
                noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)

            if (i+1 < recon_feature_idx):
                # ==> i = 1, 3, 5, 7
                out = conv2(out, latent[:, i+1], noise=noise2)
            elif i+1 == recon_feature_idx:
                # ==> i == 9
                out = feature
                out = conv2(out, latent[:, i+1], noise=noise2)
                out_ = warp_one_level(out, flow, idx, n_frames) # xxxx_debug 
                # skip = to_rgb(out_, latent[:,i+2], skip=None)
                skip = to_rgb(out_, latent[:,i+2])
            else:
                # ==> i == 11, 13, 15
                out = conv2(out, latent[:,i+1], noise=noise2)
                out_ = warp_one_level(out, flow, idx, n_frames) # xxxx_debug 
                skip = to_rgb(out_, latent[:,i+2], skip=skip)
                    
            i += 2
            image = skip
        
        image = (image + 1.0)/2.0
            
        return image.clamp(0.0, 1.0)


def upfirdn2d(input, kernel, up=1, pad=(0, 0)):
    B, C, H, W = input.shape
    H4, W4 = kernel.shape
    pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[0], pad[1]
    OH = (H * up + pad_y0 + pad_y1 - H4) + 1
    OW = (W * up + pad_x0 + pad_x1 - W4) + 1

    input = input.reshape(-1, H, W, 1)
    out = upfirdn2d_native(input, kernel, up, pad_x0, pad_x1, pad_y0, pad_y1)

    out = out.view(-1, C, OH, OW)

    return out



def upfirdn2d_native(input, kernel, up, pad_x0, pad_x1, pad_y0, pad_y1):
    # kernel.size() -- [4, 4]

    # todos.debug.output_var("input", input)
    # tensor [input] size: [512, 9, 9, 1], min: -0.696562, max: 0.689073, mean: 0.002468


    _, H, W, minor = input.shape
    H4, W4 = kernel.shape
    OH = (H * up + pad_y0 + pad_y1 - H4) + 1
    OW = (W * up + pad_x0 + pad_x1 - W4) + 1

    out = input.view(-1, H, 1, W, 1, minor)
    out = F.pad(out, [0, 0, 0, up - 1, 0, 0, 0, up - 1])
    out = out.view(-1, H * up, W * up, minor)
    out = F.pad(out, [0, 0, pad_x0, pad_x1, pad_y0, pad_y1])

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, H * up + pad_y0 + pad_y1, W * up + pad_x0 + pad_x1])

    w = torch.flip(kernel, [0, 1]).view(1, 1, H4, W4)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, OH, OW)

    out = out.permute(0, 2, 3, 1)

    # todos.debug.output_var("out", out)

    return out


def resize_flow(flow, size):
    # tensor [flow] size: [1, 2, 512, 512], min: -0.942374, max: 0.936256, mean: -0.108493
    B, C, H, W = flow.size()
    scale = size/H
    flow = F.interpolate(flow, size=(size, size), mode='bilinear', align_corners=False)
    flow *= scale # !!!
    return flow

def blend_feature(feature, flow, idx, n_frames):
    # return torch.cat([feature, feature], dim=-1)   
    # idx = 0
    # n_frames = 120

    # feature.size() -- [1, 256, 128, 128] or [1, 1, 128, 128] or [1, 128, 256, 256] ...
    size = feature.size(2)
    # pad_size = int(size / 2)
    alpha = idx / (n_frames - 1)
    
    cut_size = 0
    if feature.size(2) == 1024:
        cut_size = 3
    elif feature.size(2) == 512:
        cut_size = 2
    elif feature.size(2) == 256:
        cut_size = 1

    if not cut_size == 0:
        feature = feature[:,:,cut_size:-cut_size,cut_size:-cut_size]
        flow = flow[:,:,cut_size:-cut_size,cut_size:-cut_size]
    else:
        pass # ==> pdb.set_trace()

    ### Reflection padding for flow
    future_flow = pad_tensor(flow, mode="reflect")
    past_flow = pad_tensor(-flow, mode="reflect")
    
    ## Euler integration to get optical flow fro motion field
    future_flow = euler_integration(future_flow, idx)
    past_flow = euler_integration(past_flow, n_frames - idx - 1)
    
    ### Define importance metric Z
    Z = torch.ones(1, 1, size - 2*cut_size, size - 2*cut_size).to(flow.device)
    # todos.debug.output_var("Z", Z)
    # print("-" * 80)
    # blend_feature size = 1024, cut_size=3 -------
    # tensor [Z] size: [1, 1, 1018, 1018], min: 1.0, max: 1.0, mean: 1.0

    future_Z = pad_tensor(Z, mode="reflect") * (1.0 - alpha)
    past_Z = pad_tensor(Z, mode="reflect") * 1.0 * alpha
    
    ### Pad feature, and get segmentation mask for feature and flow regions
    feature = pad_tensor(feature, mode="reflect")
    blended = joint_splatting(feature, future_Z, future_flow, feature, past_Z, past_flow, output_size=feature.shape[-2:])
    
    return blended


def pad_tensor(tensor, mode="reflect", number=None):
    # cut_size = 0
    size = tensor.size(2)
    # pad_size = int(size / 4) + int(size / 8) + cut_size
    pad_size = size // 4 + size // 8
    # pad_tensor size = 254 ==> pad_size=94
    # pad_tensor size = 508 ==> pad_size=190
    # pad_tensor size = 1018 ==> pad_size=381
    pad = (pad_size, pad_size, pad_size, pad_size)
    return F.pad(tensor, pad, "reflect")   

def crop_padded_tensor(padded_tensor, size):
    padded_size = padded_tensor.size(2) - size
    start_idx = padded_size // 2
    end_idx = start_idx + size
    
    cropped_tensor = padded_tensor[:,:,start_idx:end_idx, start_idx:end_idx]
    return cropped_tensor


def warp_one_level(out, flow, idx, n_frames):
    orig_size = out.size(2)
    flow = resize_flow(flow, out.size(2))
    
    out = blend_feature(out, flow, idx, n_frames)

    out = crop_padded_tensor(out, orig_size)
    return out

def euler_integration(motion, frame_index):
    """
    Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

    :param motion: The Eulerian motion field to be integrated.
    :param frame_index: The number of times the motion field should be integrated.
    :return: The displacement map resulting from repeated integration of the motion field.
    """
    # tensor [motion] size: [1, 2, 224, 224], min: -0.024374, max: 0.242733, mean: 0.046427
    # frame_index = 0

    B, C, H, W = motion.shape
    y, x = torch.meshgrid([torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)])
    coord = torch.stack([x, y], dim=0).to(motion.device) # size() -- [2, 224, 224]
    d_coord = coord.clone()

    displacements = torch.zeros(1, 2, H, W).to(motion.device)
    invalid_mask = torch.zeros(1, H, W).bool().to(motion.device)
    flow =  motion[0]
    for id in range(1, frame_index + 1):
        d_r = torch.round(d_coord[0]).long()
        d_c = torch.round(d_coord[1]).long()
        d_coord += flow[:, d_c, d_r]

        out_of_bounds_x = torch.logical_or(d_coord[0] > (W - 1), d_coord[0] < 0)
        out_of_bounds_y = torch.logical_or(d_coord[1] > (H - 1), d_coord[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # invalid_mask.size() -- [1, 224, 224]
        # invalid_mask.expand_as(d_coord).size() -- [2, 224, 224]
        updated_mask = invalid_mask.expand_as(d_coord)
        d_coord[updated_mask] = coord[updated_mask]
        displacements = (d_coord - coord).unsqueeze(0)

    # todos.debug.output_var("displacements", displacements)
    # print("-" * 80)

    # frame_index = 0
    # tensor [motion] size: [1, 2, 442, 442], min: -0.49711, max: 0.086481, mean: -0.093155
    # tensor [d_coord] size: [2, 442, 442], min: 0.0, max: 441.0, mean: 220.500015
    # tensor [displacements] size: [1, 2, 442, 442], min: 0.0, max: 0.0, mean: 0.0
    # --------------------------------------------------------------------------------
    # frame_index = 119
    # tensor [motion] size: [1, 2, 888, 888], min: -0.232555, max: 0.999254, mean: 0.186372
    # tensor [d_coord] size: [2, 888, 888], min: 0.0, max: 887.0, mean: 443.499969
    # tensor [displacements] size: [1, 2, 888, 888], min: -2.862347, max: 99.450378, mean: 19.937889
    # --------------------------------------------------------------------------------

    return displacements

backwarp_tenGrid = {}
def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)




def joint_splatting(feature_map1, weights1, flow1,
                    feature_map2, weights2, flow2, 
                    output_size=None):
        
    assert (feature_map1.shape == feature_map2.shape)
    assert (flow1.shape == flow2.shape)
    assert (feature_map1.shape[-2:] == flow1.shape[-2:])

    # tensor [flow1] size: [1, 2, 224, 224], min: 0.0, max: 0.0, mean: 0.0
    # tensor [flow2] size: [1, 2, 224, 224], min: -24.74115, max: 0.575409, mean: -5.035127

    # pdb.set_trace()
    # todos.debug.output_var("flow1", flow1)
    # todos.debug.output_var("flow2", flow2)

    flow2_offset = flow2.clone().cuda()
    flow2_offset[:, 0, :, :] -= feature_map1.shape[-1]

    flow = torch.cat([flow1, flow2_offset], dim=-1)
    feature_map = torch.cat([feature_map1, feature_map2], dim=-1)
    blending_weights = torch.cat([weights1, weights2], dim=-1)

    result_softsplat = FunctionSoftsplat(tensorInput=feature_map,
                                         tensorFlow=flow,
                                         weights=blending_weights,
                                         output_size=output_size)

    # todos.debug.output_var("feature_map", feature_map)
    # todos.debug.output_var("flow", flow)
    # todos.debug.output_var("blending_weights", blending_weights)
    # todos.debug.output_var("result_softsplat", result_softsplat)
    # print("-" * 80)
    # tensor [feature_map] size: [1, 256, 224, 448], min: -1.265544, max: 10.693622, mean: 1.426945
    # tensor [flow] size: [1, 2, 224, 448], min: -224.0, max: 24.74115, mean: -53.523159
    # tensor [blending_weights] size: [1, 1, 224, 448], min: 0.0, max: 1.0, mean: 0.5
    # tensor [result_softsplat] size: [1, 256, 224, 224], min: -1.265544, max: 10.693622, mean: 1.426945
    # --------------------------------------------------------------------------------

    return result_softsplat

def FunctionSoftsplat(tensorInput, tensorFlow, weights, output_size=None):
    assert (weights is None or weights.shape[1] == 1)
    tensorInput = torch.cat([tensorInput * weights, weights], 1)
    tensorOutput = _FunctionSoftsplat.apply(tensorInput, tensorFlow)

    # # xxxx_3333
    # tensorOutput = softsplat(tensorInput[0].cpu().numpy(), tensorFlow[0].cpu().numpy())
    # tensorOutput = torch.from_numpy(tensorOutput).cuda().unsqueeze(0)

    tenSplattedMetric = tensorOutput[:, -1:, :, :]
    tenSplattedMetric[tenSplattedMetric == 0] = 1
    tensorOutput = tensorOutput[:, :-1, :, :] / tenSplattedMetric

    tensorOutput = tensorOutput[:, :, :output_size[0], :output_size[1]]

    # tensor [tensorInput] size: [1, 1, 888, 1776], min: 1.0, max: 1.0, mean: 1.0
    # tensor [tensorFlow] size: [1, 2, 888, 1776], min: -888.0, max: 99.450378, mean: -212.031052
    # tensor [weights] size: [1, 1, 888, 1776], min: 0.0, max: 1.0, mean: 0.5
    # output_size is tuple: len = 2
    #     [item] value: '888'
    #     [item] value: '888'
    # tensor [tensorOutput] size: [1, 1, 888, 888], min: 1.0, max: 1.0, mean: 1.0
    # --------------------------------------------------------------------------------
    # tensor [tensorInput] size: [1, 32, 1780, 3560], min: -1.075294, max: 3.646587, mean: 0.008901
    # tensor [tensorFlow] size: [1, 2, 1780, 3560], min: -1780.0, max: 198.96228, mean: -424.981659
    # tensor [weights] size: [1, 1, 1780, 3560], min: 0.0, max: 1.0, mean: 0.5
    # output_size is tuple: len = 2
    #     [item] value: '1780'
    #     [item] value: '1780'
    # tensor [tensorOutput] size: [1, 32, 1780, 1780], min: -1.075294, max: 3.646587, mean: 0.008901
    # --------------------------------------------------------------------------------

    return tensorOutput


kernel_Softsplat_updateOutput = '''
    extern "C" __global__ void kernel_Softsplat_updateOutput(
        const int n,
        const float* input,
        const float* flow,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX = ( intIndex                                                    ) % SIZE_3(output);

        float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(dblOutputX));
        int intNorthwestY = (int) (floor(dblOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (intSoutheastY) - dblOutputY   );
        float dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - dblOutputY   );
        float dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * (dblOutputY    - (float) (intNortheastY));
        float dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * (dblOutputY    - (float) (intNorthwestY));

        if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) * dblNorthwest);
        }

        if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) * dblNortheast);
        }

        if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) * dblSouthwest);
        }

        if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) * dblSoutheast);
        }
    } }
'''



def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))

    while True:
        objectMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), '(' + str.join('+', strIndex) + ')')

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')

    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


class _FunctionSoftsplat(torch.autograd.Function):
    @staticmethod
    def forward(self, input, flow):
        # ==> pdb.set_trace()
        self.save_for_backward(input, flow)

        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

        assert (intFlowDepth == 2)
        assert (intInputHeight == intFlowHeight)
        assert (intInputWidth == intFlowWidth)

        assert (input.is_contiguous() == True)
        assert (flow.is_contiguous() == True)

        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])

        if input.is_cuda == True:
            n = output.nelement()
            cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {
                'input': input,
                'flow': flow,
                'output': output
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input.data_ptr(), flow.data_ptr(), output.data_ptr()]
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # todos.debug.output_var("input",input)
        # todos.debug.output_var("flow",flow)
        # todos.debug.output_var("output",output)
        # tensor [input] size: [1, 257, 224, 448], min: -1.265544, max: 10.693622, mean: 0.712642
        # tensor [flow] size: [1, 2, 224, 448], min: -248.74115, max: 0.575409, mean: -58.517563
        # tensor [output] size: [1, 257, 224, 448], min: -1.265544, max: 10.693622, mean: 0.712642

        return output


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))

    def forward(self, input):
        return fused_leaky_relu(input, self.bias)


def fused_leaky_relu(input, bias):
    B, C, H, W = input.size()
    bbb = input + bias.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    return F.leaky_relu(bbb, 0.2) * (2 ** 0.5)

