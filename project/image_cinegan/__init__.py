"""Image/Video Patch Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021-2024(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import cinegan, encoder
import numpy as np

import todos
import pdb

def get_trace_model():
    """Create model."""

    seed = 240  # pick up a random number
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = cinegan.Generator()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    return model, device



def get_cinegan_model():
    """Create model."""

    seed = 240  # pick up a random number
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = cinegan.Generator()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")

    # # expamples = torch.randn(1, 4, 512, 512).to(device).clamp(0.0, 1.0)
    # # model = torch.jit.trace(model, expamples)

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_cinegan.torch"):
    #     model.save("output/image_cinegan.torch")

    return model, device

def get_encoder_model(device):
    """Create model."""

    model = encoder.Encoder()

    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")

    # # expamples = torch.randn(1, 4, 512, 512).to(device).clamp(0.0, 1.0)
    # # model = torch.jit.trace(model, expamples)

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_cineenc.torch"):
    #     model.save("output/image_cineenc.torch")

    return model



def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    cinegan_model, device = get_cinegan_model()
    encoder_model = get_encoder_model(device)

    # load files
    image_filenames = todos.data.load_files(input_files)

    # get image, mask, motion file names
    image_filename = ''
    mask_filename = ''
    motion_filename = ''
    for fn in image_filenames:
        if not '_mask.' in fn:
            image_filename = fn
        if '_mask.' in fn:
            mask_filename = fn
    motion_filename = image_filename.replace(".png", "_motion.npy")

    image_tensor = todos.data.load_tensor(image_filename).to(device)
    mask_tensor = todos.data.load_tensor(mask_filename).to(device)[:, 0:1, :, :]
    motion_tensor = torch.from_numpy(np.load(motion_filename)).to(device)

    with torch.no_grad():
        latent, feature = encoder_model(image_tensor)

    # todos.debug.output_var("latent", latent)
    # todos.debug.output_var("feature", feature)
    # tensor [latent] size: [1, 18, 512], min: -15.748751, max: 10.971325, mean: 0.022463
    # tensor [feature] size: [1, 256, 128, 128], min: -4.126172, max: 4.97429, mean: 0.400132
    
    up_flow = cinegan.resize_flow(motion_tensor, 1024)
    up_flow *= mask_tensor
    # todos.debug.output_var("up_flow", up_flow)
    # tensor [up_flow] size: [1, 2, 1024, 1024], min: -0.942374, max: 0.936256, mean: -0.108493

    n_frames = 120
    latents = []
    for _ in range(n_frames):
        latents.append(latent) # latent.size() -- [1, 18, 512]


    # start predict
    progress_bar = tqdm(total=len(latents))
    for index, latent in enumerate(latents):
        progress_bar.update(1)

        with torch.no_grad():    
            result = cinegan_model(latent,
                        feature=feature,
                        idx=index,
                        n_frames=n_frames,
                        flow=up_flow
                    )
        # todos.debug.output_var("result", result)
        # tensor [result] size: [1, 3, 1024, 1024], min: -0.278078, max: -0.139095, mean: -0.248583            
        output_tensor = result*mask_tensor + image_tensor * (1 - mask_tensor)

        output_file = f"{output_dir}/{index:06d}.png"
        todos.data.save_tensor([output_tensor], output_file)

    todos.model.reset_device()
