import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.joint_splatting import joint_splatting
import todos
import pdb

def euler_integration(motion, destination_frame):
    """
    Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

    :param motion: The Eulerian motion field to be integrated.
    :param destination_frame: The number of times the motion field should be integrated.
    :return: The displacement map resulting from repeated integration of the motion field.
    """

    # tensor [motion] size: [1, 2, 224, 224], min: -0.024374, max: 0.242733, mean: 0.046427
    # destination_frame = 0

    assert (motion.dim() == 4)
    b, c, height, width = motion.shape
    assert (b == 1), 'Function only implemented for batch = 1'
    assert (c == 2), f'Input motion field should be Bx2xHxW. Given tensor is: {motion.shape}'

    y, x = torch.meshgrid(
        [torch.linspace(0, height - 1, height, device='cuda'),
         torch.linspace(0, width - 1, width, device='cuda')])
    coord = torch.stack([x, y], dim=0).long()
    destination_coords = coord.clone().float().cuda()
    # tensor [destination_coords] size: [2, 442, 442], min: 0.0, max: 441.0, mean: 220.500015

    # print(f"destination_frame = {destination_frame}")
    # todos.debug.output_var("motion", motion)
    # todos.debug.output_var("destination_coords", destination_coords)

    motion = motion.cuda()

    displacements = torch.zeros(1, 2, height, width, device='cuda')
    invalid_mask = torch.zeros(1, height, width, device='cuda').bool()

    for frame_id in range(1, destination_frame + 1):
        # print(f"frame_id = {frame_id}")
        destination_coords = destination_coords + motion[0][:, torch.round(destination_coords[1]).long(),
                                                  torch.round(destination_coords[0]).long()]
        out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
        out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
        destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
            invalid_mask.expand_as(destination_coords)].float()

        displacements = (destination_coords - coord.float()).unsqueeze(0)

    # todos.debug.output_var("displacements", displacements)
    # print("-" * 80)

    # destination_frame = 0
    # tensor [motion] size: [1, 2, 442, 442], min: -0.49711, max: 0.086481, mean: -0.093155
    # tensor [destination_coords] size: [2, 442, 442], min: 0.0, max: 441.0, mean: 220.500015
    # tensor [displacements] size: [1, 2, 442, 442], min: 0.0, max: 0.0, mean: 0.0
    # --------------------------------------------------------------------------------
    # destination_frame = 119
    # tensor [motion] size: [1, 2, 888, 888], min: -0.232555, max: 0.999254, mean: 0.186372
    # tensor [destination_coords] size: [2, 888, 888], min: 0.0, max: 887.0, mean: 443.499969
    # tensor [displacements] size: [1, 2, 888, 888], min: -2.862347, max: 99.450378, mean: 19.937889
    # --------------------------------------------------------------------------------

    return displacements


def pad_tensor(tensor, mode="reflect", number=None):
    # cut_size = 0
    size = tensor.size(2)
    # pad_size = int(size / 4) + int(size / 8) + cut_size
    pad_size = size // 4 + size // 8

    # pad_tensor size = 254 ==> pad_size=94
    # pad_tensor size = 508 ==> pad_size=190
    # pad_tensor size = 1018 ==> pad_size=381
    pad = (pad_size, pad_size, pad_size, pad_size)

    if mode=="reflect":
        # pad = nn.ReflectionPad2d(pad_size)
        # padded_tensor = pad(tensor)
        padded_tensor = F.pad(tensor, pad, "reflect")   
    elif mode=="constant": # False
        pdb.set_trace()
        padded_tensor = F.pad(tensor, pad, "constant", number)

    return padded_tensor


def crop_padded_tensor(padded_tensor, size):
    padded_size = padded_tensor.size(2) - size
    start_idx = padded_size // 2
    end_idx = start_idx + size
    
    cropped_tensor = padded_tensor[:,:,start_idx:end_idx, start_idx:end_idx]
    return cropped_tensor


def resize_mask(mask, size):
    # tensor [mask] size: [1, 1, 512, 512], min: 0.0, max: 1.0, mean: 0.618692
    B, C, H, W = mask.size()
    return F.interpolate(mask, size=(size, size), mode='bilinear', align_corners=False)



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
    future_flow = pad_tensor(flow, mode="reflect").float().cuda()
    past_flow = pad_tensor(-flow, mode="reflect").float().cuda()
    
    ## Euler integration to get optical flow fro motion field
    future_flow = euler_integration(future_flow, idx)
    past_flow = euler_integration(past_flow, n_frames - idx - 1)
    
    ### Define importance metric Z
    Z = torch.ones(1, 1, size - 2*cut_size, size - 2*cut_size)
    # todos.debug.output_var("Z", Z)
    # print("-" * 80)
    # blend_feature size = 1024, cut_size=3 -------
    # tensor [Z] size: [1, 1, 1018, 1018], min: 1.0, max: 1.0, mean: 1.0

    future_Z = Z.float().cuda()
    future_Z = pad_tensor(future_Z, mode="reflect").float().cuda() * (1 - alpha)
    
    past_Z = Z.float().cuda()
    past_Z = pad_tensor(past_Z, mode="reflect").float().cuda() * alpha
    
    ### Pad feature, and get segmentation mask for feature and flow regions
    feature = pad_tensor(feature, mode="reflect").float().cuda()
    blended = joint_splatting(feature, future_Z, future_flow, feature, past_Z, past_flow, output_size=feature.shape[-2:])
    
    out = blended.cuda()
    return out.type(torch.float)


def warp_one_level(out, flow, idx, n_frames):
    orig_size = out.size(2)
    flow = resize_flow(flow, out.size(2))
    
    out = blend_feature(out, flow, idx, n_frames)

    out = crop_padded_tensor(out, orig_size)
    return out
