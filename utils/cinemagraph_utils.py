import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.joint_splatting import joint_splatting #, backwarp
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

    motion = motion.cuda()

    displacements = torch.zeros(1, 2, height, width, device='cuda')
    invalid_mask = torch.zeros(1, height, width, device='cuda').bool()

    for frame_id in range(1, destination_frame + 1):
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

    return displacements


def pad_tensor(tensor, mode="reflect", number=None):
    cut_size = 0
    size = tensor.size(2)
    pad_size = int(size / 4) + int(size / 8) + cut_size
    pad = (pad_size, pad_size, pad_size, pad_size)

    if mode=="reflect":
        pad = nn.ReflectionPad2d(pad_size)
        padded_tensor = pad(tensor)    
    elif mode=="constant": # False
        padded_tensor = F.pad(tensor, pad, "constant", number)

    return padded_tensor


def crop_padded_tensor(padded_tensor, size):
    padded_size = padded_tensor.size(2) - size
    start_idx = int(padded_size/2)
    end_idx = start_idx + size
    
    cropped_tensor = padded_tensor[:,:,start_idx:end_idx, start_idx:end_idx]
    return cropped_tensor


def resize_feature(feature, size):
    if feature.size(2) < size:
        while feature.size(2) < size:
            up_height = feature.shape[2] * 2
            up_width = feature.shape[3] * 2

            feature = nn.functional.interpolate(feature, size=(up_height, up_width), mode='bilinear', align_corners=False)
    
    elif feature.size(2) > size:
        down_height = int(feature.shape[2] / 2)
        down_width = int(feature.shape[3] / 2)
        
        feature = nn.functional.interpolate(feature, size=(down_height, down_width), mode='bilinear', align_corners=False)
        
    return feature


def resize_flow(flow, size):
    flow_size = flow.size(2)
    
    while flow.size(2) != size:
        
        mode = 'downsample'
    
        if flow_size > size:
            height = int(flow.size(2) / 2)
            width = int(flow.size(3) / 2)
            
        elif flow_size < size:
            height = int(flow.size(2) * 2)
            width = int(flow.size(3) * 2)
            mode = 'upsample'
            
        flow = nn.functional.interpolate(flow, size=(height, width), mode='bilinear', align_corners=False)
        
        if mode == 'downsample':
            flow /= 2.0
        elif mode == 'upsample':
            flow *= 2.0

    return flow


def blend_feature(feature, flow, idx, n_frames):
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
    past_flow = euler_integration(past_flow, n_frames-idx-1)
    
    ### Define importance metric Z
    Z = torch.ones(1, 1, size - 2*cut_size, size - 2*cut_size)
    future_Z = Z.float().cuda()
    future_Z = pad_tensor(future_Z, mode="reflect").float().cuda() * (1 - alpha)
    
    past_Z = Z.float().cuda()
    past_Z = pad_tensor(past_Z, mode="reflect").float().cuda() * alpha
    
    ### Pad feature, and get segmentation mask for feature and flow regions
    feature = pad_tensor(feature, mode="reflect").float().cuda()
    blended = joint_splatting(feature, future_Z, future_flow,
                              feature, past_Z, past_flow,
                              output_size=feature.shape[-2:]
                            )
    # pdb.set_trace()
    
    out = blended.cuda()
    return out.type(torch.float)


def warp_one_level(out, flow, idx, n_frames):
    orig_size = out.size(2)
    flow = resize_flow(flow, out.size(2))
    
    out = blend_feature(out, flow, idx, n_frames)
    # out = feature_inpaint_conv(out, flow, idx, n_frames) # xxxx_3333
    out = crop_padded_tensor(out, orig_size)
    return out
    
# def feature_inpaint_conv(feature, flow, idx, n_frames):
#     size = feature.size(2)
#     pad_size = int(size / 2)
    
#     bn_feature = torch.ones(1, 1, flow.size(2), flow.size(3))
#     warped_bn_feature = blend_feature(bn_feature, flow, idx, n_frames)
#     blank_mask = torch.where(warped_bn_feature==0, 1, 0)
    
#     full_mask = 1 - blank_mask

#     if blank_mask.max() == 1.:
#         # ==> pdb.set_trace()
#         ### make kernel
#         weights = torch.ones(1, 1, 7, 7).cuda() / 49
#         filtered = torch.zeros_like(feature)

#         for i in range(feature.size(1)):
#             feat_channel = feature[:,i,:,:].unsqueeze(0)
#             feat_filtered = F.conv2d(feat_channel, weights, padding=3)
#             filtered[:,i,:,:] = feat_filtered
    
#         ### Apply mask
#         output = blank_mask*filtered + full_mask*feature
#     else:
#         # ==> pdb.set_trace()
#         output = feature
    
#     return output