import torch
from torchvision import transforms
from PIL import Image
import todos
import pdb

device = torch.device("cuda")

def gan_inversion(encoder, img):
    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor_img = img_to_tensor(Image.fromarray(img)).unsqueeze(0).to(device)

    output = encoder.test(img=tensor_img)
    # tensor [tensor_img] size: [1, 3, 1024, 1024], min: -1.0, max: 1.0, mean: -0.032175
    # output is list: len = 3
    #     tensor [item] size: [1, 3, 1024, 1024], min: -1.077568, max: 1.35515, mean: -0.034574
    #     tensor [item] size: [1, 18, 512], min: -15.748751, max: 10.971326, mean: 0.022463
    #     tensor [item] size: [1, 256, 128, 128], min: -4.126171, max: 4.974294, mean: 0.400132

    feature = output.pop()
    latent = output.pop()
    result = output.pop()
    return result, latent, feature   