#%%
import torch
from PIL import Image

import itertools
from torchvision import datasets, transforms
#%%
tensor2PIL=transforms.ToPILImage()
PIL2tensor=transforms.ToTensor()
mapping={(0, 0, 0):0,
         (255, 255, 255):1}#(0, 0, 0): background, (255, 255, 255): teeth

# 두장인풋으로너어서
def mask2label(mask_ori):## RGB tensor
    
    mask_bg = mask_ori.mul(-1.0).add(1.0)
    label = torch.concat([mask_bg, mask_ori], dim=0)
            
    return label ## tensor