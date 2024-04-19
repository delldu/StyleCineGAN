import torch
from tqdm import tqdm
import mediapy as mp
import numpy as np
import os
import cv2

from PIL import Image

import warnings
warnings.filterwarnings(action='ignore')


device = torch.device('cuda')


from option import Options
from utils.model_utils import load_encoder, load_stylegan2
from utils.ip_utils import read_image, resize_tensor, to_numpy, to_tensor #, make_sg2_features_fs #, make_sg2_features
from utils.utils import gan_inversion #,clip_img, predict_mask
from utils.flow_utils import flow2img # , z_score_filtering
from utils.cinemagraph_utils import feature_inpaint, resize_flow, resize_feature
from torchvision.transforms import GaussianBlur

import todos
import pdb


if __name__ == "__main__":
    
    # load options -----------------------------------------------------------------------------------
    opts = Options().parse()
    opts.device = device
    
    # make save directory ---------------------------------------------------------------------------
    os.makedirs(f"{opts.save_dir}", exist_ok=True)
    
    # Namespace(encoder_type='fs', encoder_ckpt='./pretrained_models', 
    #     sg2_ckpt='./pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt', 
    #     datasetgan_dir='./pretrained_models/datasetGAN', 
    #     eulerian_dir='./pretrained_models/epoch-20-feature_encoder.pth', 
    #     flownet_dir='./pretrained_models', flownet_mode='sky', feature_level=9, 
    #     ir_se50='./pretrained_models/model_ir_se50.pth', 
    #     moco='./pretrained_models/moco_v2_800ep_pretrain.pt', 
    #     img_path='./samples/0002268', style_path=None, save_dir='./results', 
    #     n_frames=120, channel_multiplier=2, is_optim=False, optim_step=3000, 
    #     optim_threshold=0.005, optim_params='feat', initial_lr=0.1, random_sample=False, 
    #     style_interp=False, style_extrapolate_scale=2.0, mode='full', recon_feature_idx=9, 
    #     warp_feature_idx=9, vis='True', image_inpainting=False, no_image_composit=False, device=device(type='cuda'))

    # from PIL import Image
    # dotarrow_tensor = read_image(f"dotarrow.png", dim=128)
    # dotarrow_tensor = torch.where(dotarrow_tensor > 0.9, torch.zeros_like(dotarrow_tensor), dotarrow_tensor)
    # # dotarrow_tensor = torch.where(dotarrow_tensor < 0.1, torch.zeros_like(dotarrow_tensor), dotarrow_tensor)
    # T = GaussianBlur(kernel_size=(5, 5), sigma=(5, 5))
    # dotarrow_tensor = T(dotarrow_tensor)
    # todos.debug.output_tensor(dotarrow_tensor)


    
    # load models ------------------------------------------------------------------------------------
    sg2      = load_stylegan2(opts.sg2_ckpt)
    # sg2 -- DataParallel((module): Generator(...))
    # opts.sg2_ckpt -- './pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt'

    encoder  = load_encoder(opts.encoder_ckpt, encoder_type=opts.encoder_type, recon_idx=opts.recon_feature_idx).to(device)
    # encoder -- Trainer((enc): fs_encoder_v2(...))
    # opts.encoder_ckpt -- './pretrained_models'
    # opts.encoder_type -- 'fs', opts.recon_feature_idx -- 10    

    # read images ------------------------------------------------------------------------------------
    basename_input = (opts.img_path).split("/")[-1]
    torch_input = read_image(f"{opts.img_path}/{basename_input}.png", dim=1024, is_square=True, return_tensor=False)
    # './samples/0002268/0002268.png'
    # tensor [torch_input] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: 0.110929
    
    if opts.style_path: # None
        torch_style = read_image(opts.style_path, is_square=True, return_tensor=True)
        basename_style = os.path.basename(opts.style_path).split('.')[0]
    
    
    # invert the image ------------------------------------------------------------------------------
    with torch.no_grad():
        tensor_recon, latent, feature = gan_inversion(encoder, torch_input, model=opts.encoder_type)
    # tensor [tensor_recon] size: [1, 3, 1024, 1024], min: -1.422519, max: 1.449161, mean: 0.109709
    # tensor [latent] size: [1, 18, 512], min: -15.776832, max: 11.293081, mean: 0.022378
    # tensor [feature] size: [1, 256, 128, 128], min: -4.576892, max: 4.600057, mean: 0.402872

    if opts.style_path: # None
        pdb.set_trace()
        mean_latent = sg2.mean_latent(10000)
        mean_latent = mean_latent.detach()
        torch_style = to_tensor(torch_style)
        
        # invert the style image via optimization
        from optimize_latent import OptimizeLatent
        optim_latents = OptimizeLatent(opts, sg2, threshold=opts.optim_threshold)
        tensor_recon_style, latent_style = optim_latents.optimize_latent(torch_style, mean_latent, step=1500,
                                                                         initial_lr=opts.initial_lr,
                                                                         optim_params='latent')
    torch_input = to_tensor(torch_input)
        
    # visualize inversion results -------------------------------------------------------------------
    if opts.vis == "True": # True
        img_list = [torch_input, tensor_recon]
        if opts.style_path: # None
            img_list += [torch_style, tensor_recon_style]
        
        img_list = [to_numpy(resize_tensor(img, [512,512]))[0] for img in img_list]
        img_list = np.concatenate(img_list, axis=1)
        mp.write_image(f"{opts.save_dir}/{basename_input}_recon.png", img_list)
    
    
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
    if opts.style_interp: # False
        alphas = get_alphas(opts.n_frames, opts.style_extrapolate_scale)

    frames = []
    with torch.no_grad():
        latents = []

        # get all latents
        if opts.style_interp: # False
            for alpha in alphas:
                input_latent = latent_style * alpha +  latent * (1 - alpha)
                latents.append(input_latent)
        else: # True
            for alpha in alphas:
                latents.append(latent) # latent.size() -- [1, 18, 512]
    
        # todos.debug.output_var("latents", latents)
        # latents is list: len = 120
        #     tensor [item] size: [1, 18, 512], min: -15.776832, max: 11.293081, mean: 0.022378
        # tensor [feature] size: [1, 256, 128, 128], min: -4.576892, max: 4.600057, mean: 0.402872
        # tensor [flow] size: [1, 2, 512, 512], min: -0.942374, max: 0.936256, mean: -0.108493

        # up_mask = resize_feature(mask.float(), 1024)
        # up_flow = resize_flow(flow, 1024)
    
        # generate frames
        pbar = tqdm(total=len(latents))
        for idx, input_latent in enumerate(latents):    
            result, _ = sg2.module.warp_blend_feature(styles=[input_latent],
                                                        feature=feature,
                                                        idx=idx,
                                                        n_frames=opts.n_frames,
                                                        flow=flow, ###!!!!!!!!!!!!!!!!###
                                                        mode=opts.mode, # 'full'
                                                        Z=None,
                                                        recon_feature_idx=opts.recon_feature_idx,
                                                        warp_feature_idx=opts.warp_feature_idx,
                                                        input_is_latent=True,
                                                        return_latents=True,
                                                        randomize_noise=False,
                                                        is_random=False
                                              )
            
            # todos.debug.output_var("result", result)
            # tensor [result] size: [1, 3, 1024, 1024], min: -0.278078, max: -0.139095, mean: -0.248583            
            
            up_mask = resize_feature(mask.float(), 1024)
            up_flow = resize_flow(flow, 1024)
            
            if opts.image_inpainting: # False
                result = feature_inpaint(result, up_flow, idx, opts.n_frames)
                
            if not opts.no_image_composit: # True
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
        
