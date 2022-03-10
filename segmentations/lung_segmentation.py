'''
Author: Baoyun Peng
Date: 2022-03-10 09:06:11
LastEditTime: 2022-03-10 11:14:09
Description: Lung Segmentation using UNet

'''

import torch
import torchvision
from PIL import Image
import sys
import pdb
from src.models import PretrainedUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
)

def model_init(weight_path='../checkpoints/segment-unet-6v.pt'):
    global device, unet
    unet.load_state_dict(torch.load(weight_path, map_location="cpu"))
    unet.to(device)
    unet.eval()

def segment(img_path):
    origin = Image.open(img_path).convert("P")
    origin = torchvision.transforms.functional.resize(origin, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
    with torch.no_grad():
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = unet(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
        origin = origin[0].to("cpu")
        out = out[0].to("cpu")
    out =  torchvision.transforms.functional.to_pil_image(out.float())
    return out

def main():
    in_prefix  = '../baks/'
    out_prefix = '../baks/mask'
    # import matplotlib.pyplot as plt
    imglist = sys.argv[1]
    model_init()
    import os
    with open(imglist, 'r') as fin:
        for line in fin.readlines():
            img_name = os.path.join(in_prefix, line.strip())
            out = segment(img_name)
            out.save( os.path.join(out_prefix, line.strip()) )

if __name__ == '__main__':
    main()