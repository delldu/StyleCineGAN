
import torch
import torch.nn as nn
import mediapy as mp
from PIL import Image

device = "cuda"

def read_image(img_dir, dim=1024, is_square=True, return_tensor=True):
    np_img = mp.read_image(img_dir)
    
    # crop to square
    if is_square:
        np_img = crop_image(np_img)
    
    # resize
    np_img = mp.resize_image(np_img, (dim, dim))

    # to tensor
    if return_tensor:
        return to_tensor(np_img)
    return np_img


def resize_tensor(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


def to_tensor(x, norm=True):
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    if norm:
        x = x / 127.5 - 1
    return x


def to_numpy(x, norm=True):
    if norm:
        x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().detach().numpy()
    return x

def crop_image(x):
    # crop image to square
    H, W, _ = x.shape
    
    if H != W:
        if H < W:
            pos1 = (W - H) / 2
            pos2 = (W + H) / 2

            pos1 = int(pos1)
            pos2 = int(pos2)
            
            x = x[:,pos1:pos2,:]
        elif H > W:
            pos1 = (H - W) / 2
            pos2 = (H + W) / 2
            
            pos1 = int(pos1)
            pos2 = int(pos2)

            x = x[pos1:pos2,:]
    return x

