#%%
import torch
from PIL import Image

import itertools
from torchvision import datasets, transforms
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
#%%
tensor2PIL=transforms.ToPILImage()
PIL2tensor=transforms.ToTensor()
mapping={(0, 0, 0):0,
         (255, 255, 255):1}#(0, 0, 0): background, (255, 255, 255): teeth
#%%
def mask2label(mask_ori):## RGB tensor
    mask=PIL2tensor(mask_ori)## label별로 색이 다른 rgb
    pix=mask.load()
    
    label = torch.zeros(2, 512, 512, dtype=torch.float)
    for k in mapping:
        v=mapping.get(k)
        # w:가로 h: 세로
        for w, h in itertools.product(range(mask.size[0]),range(mask.size[1])):
            label[v][h][w]=(pix[w, h]==k)
    return label ## tensor
#%%
mask=Image.open('/home/h/Desktop/data/4-4/t_label/t_label4-4_001.png').convert('RGB')
# mask.show()
# print(PIL2tensor(mask))
mask_t=PIL2tensor(mask)
print(mask.size)# pillow 가로 세로-> w h
print(mask.load()[2000, 1400])
print(mask.load()[1400, 2000])
# %%
label = torch.zeros(2, 512, 512, dtype=torch.float)
# tensor2PIL(label[0]).show()
# tensor2PIL(label[1]).show()
tensor2PIL(label).show()
#%%
pix=mask.load()

for k in mapping:
    v=mapping.get(k)
    # w:가로 h: 세로
    for w, h in itertools.product(range(mask.size[0]),range(mask.size[1])):
        label[v][h][w]=(pix[w, h]==k)
#%%
tensor2PIL(label).show()
tensor2PIL(label[1]).show()
tensor2PIL(label[0]).show()
# %%
label.show()

