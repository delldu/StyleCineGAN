import torch

from utils.softmax_splatting import FunctionSoftsplat
import todos
import pdb

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
                                         tensorMetric=blending_weights,
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

