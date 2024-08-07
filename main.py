import torch
from tqdm import tqdm
import mediapy as mp
import numpy as np
import os

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda')

from option import Options
from utils.model_utils import load_encoder, load_stylegan2
from utils.ip_utils import read_image, resize_tensor, to_numpy, to_tensor
from utils.utils import gan_inversion
from utils.flow_utils import flow2img
from utils.cinemagraph_utils import resize_flow, resize_mask

import todos
import pdb


if __name__ == "__main__":
    
    # load options -----------------------------------------------------------------------------------
    opts = Options().parse()
    opts.device = device
    
    # make save directory ---------------------------------------------------------------------------
    os.makedirs(f"{opts.save_dir}", exist_ok=True)
    
    # load models ------------------------------------------------------------------------------------
    sg2      = load_stylegan2(opts.sg2_ckpt)
    # sg2 -- DataParallel((module): Generator(...))
    # opts.sg2_ckpt -- './pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt'

    # (Pdb) opts.log_path -- './pretrained_models/logs/lhq_k10'
    encoder  = load_encoder(opts.encoder_ckpt, recon_idx=opts.recon_feature_idx).to(device)
    # encoder -- Trainer((enc): fs_encoder_v2(...))
    # opts.encoder_ckpt -- './pretrained_models'
    # torch.save(encoder.state_dict(), "/tmp/e.pth") # 427M



    # read images ------------------------------------------------------------------------------------
    basename_input = (opts.img_path).split("/")[-1]
    torch_input = read_image(f"{opts.img_path}/{basename_input}.png", dim=1024, is_square=True, return_tensor=False)
    # './samples/0002268/0002268.png'
    # tensor [torch_input] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: 0.110929
    
    
    # invert the image ------------------------------------------------------------------------------
    with torch.no_grad():
        # tensor_recon, latent, feature = gan_inversion(encoder, torch_input)
        latent, feature = gan_inversion(encoder, torch_input)

    # tensor [latent] size: [1, 18, 512], min: -15.776832, max: 11.293081, mean: 0.022378
    # tensor [feature] size: [1, 256, 128, 128], min: -4.576892, max: 4.600057, mean: 0.402872

    torch_input = to_tensor(torch_input)
        
    # visualize inversion results -------------------------------------------------------------------
    # if opts.vis == "True": # True
    #     img_list = [torch_input, tensor_recon]
    #     img_list = [to_numpy(resize_tensor(img, [512,512]))[0] for img in img_list]
    #     img_list = np.concatenate(img_list, axis=1)
    #     mp.write_image(f"{opts.save_dir}/{basename_input}_recon.png", img_list)
    
    
    # load flow ----------------------------------------------------------------------------------
    print(f"\n>>> Loading Flow...")
    flow = np.load(f"{opts.img_path}/{basename_input}_motion.npy") # './samples/0002268/0002268_motion.npy'
    flow = torch.from_numpy(flow).to(device)
    # tensor [flow] size: [1, 2, 512, 512], min: -0.942374, max: 0.936256, mean: -0.108493
    # flow = -1.0 * flow
    print(">>> Done -------------------------- \n")
    
    # load mask ----------------------------------------------------------------------------------
    print("\n>>> Loading Mask...")
    mask = mp.read_image(f"{opts.img_path}/{basename_input}_mask.png")
    mask = mp.resize_image(mask, (512, 512))[:,:,0] / 255
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    # tensor [mask] size: [1, 1, 512, 512], min: 0.0, max: 1.0, mean: 0.618692
    flow *= mask
    print(">>> Done -------------------------- \n")
        
    
    # visualize optical flow---------------------------------------------------------------
    if opts.vis == "True": # True
        flows = [flow2img(flow[0].permute(1, 2, 0).cpu().detach().numpy())]
        flows = np.concatenate(flows, axis=1)
        mp.write_image(f"{opts.save_dir}/{basename_input}_flow.png", flows)
    
    
    # generate cinemagraph ---------------------------------------------------------------------------   
    print("\n>>> Generating Cinemagraph...")
    
    # get alphas
    alphas = torch.ones(opts.n_frames, 1).to(device)

    frames = []
    with torch.no_grad():
        latents = []

        for alpha in alphas:
            latents.append(latent) # latent.size() -- [1, 18, 512]
    
        # todos.debug.output_var("latents", latents)
        # latents is list: len = 120
        #     tensor [item] size: [1, 18, 512], min: -15.776832, max: 11.293081, mean: 0.022378
        # tensor [feature] size: [1, 256, 128, 128], min: -4.576892, max: 4.600057, mean: 0.402872
        # tensor [flow] size: [1, 2, 512, 512], min: -0.942374, max: 0.936256, mean: -0.108493

        up_mask = resize_mask(mask.float(), 1024)
        up_flow = resize_flow(flow, 1024)
    
        # generate frames
        pbar = tqdm(total=len(latents))
        for idx, input_latent in enumerate(latents):    
            result = sg2.module.forward(
                        input_latent,
                        feature=feature,
                        idx=idx,
                        n_frames=opts.n_frames,
                        flow=flow, ###!!!!!!!!!!!!!!!!###
                    )
            
            # todos.debug.output_var("result", result)
            # tensor [result] size: [1, 3, 1024, 1024], min: -0.278078, max: -0.139095, mean: -0.248583            
            
            result = result*up_mask + torch_input.cuda() * (1 - up_mask)
            
            result = to_numpy(result)[0]
            frames.append(np.array(result))
            
            pbar.update(1)
        pbar.close()
            
        
        # save video
        if opts.style_path: # None
            vid_name = f"{opts.save_dir}/{basename_input}_{basename_style}.mp4"
        else:
            vid_name = f"{opts.save_dir}/{basename_input}.mp4"\
            
        mp.write_video(vid_name, frames, fps=30)
        print(">>> Done -------------------------- \n")
        
