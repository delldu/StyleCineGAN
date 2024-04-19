import torch
# import cv2
# import numpy as np
from torchvision import transforms #, utils
from PIL import Image
import todos
import pdb

device = torch.device("cuda")


# def clip_img(x):
#     """Clip stylegan generated image to range(0,1)"""
#     img_tmp = x.clone()[0]
#     img_tmp = (img_tmp + 1) / 2
#     img_tmp = torch.clamp(img_tmp, 0, 1)
#     return img_tmp


def gan_inversion(encoder, img, model='fs'):
    # if model=='fs':
        
    #     img_to_tensor = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     ])
        
    #     tensor_img = img_to_tensor(Image.fromarray(img)).unsqueeze(0).to(device)

    #     output = encoder.test(img=tensor_img, return_latent=True)
    #     # tensor [tensor_img] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: -0.032175
    #     # output is list: len = 4
    #     #     tensor [item] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: -0.032175
    #     #     tensor [item] size: [1, 3, 1024, 1024], min: -1.077568, max: 1.35515, mean: -0.034574
    #     #     tensor [item] size: [1, 18, 512], min: -15.748751, max: 10.971326, mean: 0.022463
    #     #     tensor [item] size: [1, 256, 128, 128], min: -4.126171, max: 4.974294, mean: 0.400132

    #     feature = output.pop()
    #     latent = output.pop()
    #     result = output[1]
    #     return result, latent, feature    
    
    # elif model=='psp':
    #     ### Resize image to 256X256
    #     tensor_img = to_tensor(img)
    #     tensor_input = resize_tensor(tensor_img, (256,256)).cuda()        
    #     result, latent = encoder(tensor_input, resize=False, randomize_noise=False, return_latents=True)
    #     return result, latent


    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor_img = img_to_tensor(Image.fromarray(img)).unsqueeze(0).to(device)

    output = encoder.test(img=tensor_img, return_latent=True)
    # tensor [tensor_img] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: -0.032175
    # output is list: len = 4
    #     tensor [item] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: -0.032175
    #     tensor [item] size: [1, 3, 1024, 1024], min: -1.077568, max: 1.35515, mean: -0.034574
    #     tensor [item] size: [1, 18, 512], min: -15.748751, max: 10.971326, mean: 0.022463
    #     tensor [item] size: [1, 256, 128, 128], min: -4.126171, max: 4.974294, mean: 0.400132

    feature = output.pop()
    latent = output.pop()
    result = output[1]
    return result, latent, feature   