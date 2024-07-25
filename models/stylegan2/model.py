import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from models.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from utils.cinemagraph_utils import warp_one_level

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
        # ==> pdb.set_trace()

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias: # True
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            pdb.set_trace()
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation: # True
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            pdb.set_trace()
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    # def forward(self, input, style, flow=None, idx=None, n_frames=None):
    def forward(self, input, style):
        # flow = None
        # idx = None
        # n_frames = None

        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate: # True | False
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        else:
            # ==> pdb.set_trace()
            pass

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample: # False ?
            pdb.set_trace()

            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)

            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out
    

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        return out

class ToRGBUpsample(nn.Module):
    def __init__(self, in_channel, style_dim, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        # if skip is not None: # True | False
        #     # ==>pdb.set_trace()
        #     skip = self.upsample(skip)
        #     out = out + skip
        skip = self.upsample(skip)
        out = out + skip

        return out

class ToRGBUpsampleSkipNone(nn.Module):
    def __init__(self, in_channel, style_dim, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = Upsample(blur_kernel) # useless !!!
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
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
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()
        # size = 1024
        # style_dim = 512
        # n_mlp = 8
        # channel_multiplier = 2
        # blur_kernel = [1, 3, 3, 1]
        # lr_mlp = 0.01

        # self.size = size
        # self.style_dim = style_dim

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu')
            )

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
        self.conv1 = StyledConv(
            channel_512, channel_512, 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channel_512, style_dim)

        self.log_size = int(math.log(size, 2)) # 10
        self.num_layers = (self.log_size - 2) * 2 + 1 # ==> 17

        self.convs = nn.ModuleList()
        # self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = channel_512

        for layer_idx in range(self.num_layers): # self.num_layers === 17
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1): # self.log_size === 10 ==> 3, 4, 5, 6, 7, 8, 9, 10
            out_channel = channels[2 ** i]

            self.convs.append(
                StyledConv(in_channel, out_channel, 3, style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel)
            )
            # i === 7 ==> ToRGBUpsampleSkipNone            
            if i == 7:
                self.to_rgbs.append(ToRGBUpsampleSkipNone(out_channel, style_dim))
            else:
                self.to_rgbs.append(ToRGBUpsample(out_channel, style_dim))

            in_channel = out_channel

        # self.load_weights(...)
        # self.convs1 = ...
        # self.convs2 = ...
        # self.noise1 = ...
        # self.noise2 = ...
        # self.n_latent = self.log_size * 2 - 2 # self.n_latent === 18
        # pdb.set_trace()
        
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
        
        
        return image


# class ConvLayer(nn.Sequential):
#     def __init__(self,
#         in_channel,
#         out_channel,
#         kernel_size,
#         downsample=False,
#         blur_kernel=[1, 3, 3, 1],
#         bias=True,
#         activate=True,
#     ):
#         layers = []
#         if downsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) + (kernel_size - 1)
#             pad0 = (p + 1) // 2
#             pad1 = p // 2

#             layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

#             stride = 2
#             self.padding = 0
#         else:
#             stride = 1
#             self.padding = kernel_size // 2

#         layers.append(
#             EqualConv2d(
#                 in_channel,
#                 out_channel,
#                 kernel_size,
#                 padding=self.padding,
#                 stride=stride,
#                 bias=bias and not activate,
#             )
#         )

#         if activate:
#             if bias:
#                 layers.append(FusedLeakyReLU(out_channel))
#             else:
#                 layers.append(ScaledLeakyReLU(0.2))

#         super().__init__(*layers)


# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()

#         self.conv1 = ConvLayer(in_channel, in_channel, 3)
#         self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

#         self.skip = ConvLayer(
#             in_channel, out_channel, 1, downsample=True, activate=False, bias=False
#         )

#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.conv2(out)

#         skip = self.skip(input)
#         out = (out + skip) / math.sqrt(2)

#         return out

