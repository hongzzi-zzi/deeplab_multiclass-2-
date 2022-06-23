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
#%%
# def mask2label(mask_ori):## RGB tensor
#     mask=(mask_ori)## label별로 색이 다른 rgb
#     # mask=PIL2tensor(mask_ori)## label별로 색이 다른 rgb
#     label = torch.empty(2, 512, 512, dtype=torch.float)
#     for k in mapping:
#         idx=(mask==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
#         validx = (idx.sum(0) == 3)  # Check that all channels match
#         label[mapping.get(k)][validx] = torch.tensor(mapping[k], dtype=torch.long)
#     return label
def mask2label(mask_ori):## RGB tensor
    mask=tensor2PIL(mask_ori)
    pix=mask.load()
    label = torch.zeros(2, 512, 512, dtype=torch.float)
    for k in mapping:
        v=mapping.get(k)
        # w:가로 h: 세로
        for w, h in itertools.product(range(mask.size[0]),range(mask.size[1])):
            label[v][h][w]=(pix[w, h]==k)
    return label ## tensor
'''#%%
def mask2label2(mask_ori):
    ## mask: tensor
    mask=PIL2tensor(mask2RGBmask(mask_ori))## label별로 색이 다른 rgb
    label = torch.empty(512, 512, dtype=torch.long)
    for k in mapping:
        idx=(mask==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)  # Check that all channels match
        label[validx] = torch.tensor(mapping[k], dtype=torch.long)
    return label'''
# %%
'''mask_ori=PIL2tensor(Image.open('/home/h/Desktop/data/random/train/t_label/t_label1_002.png'))
mask=PIL2tensor(mask2RGBmask(mask_ori))## label별로 색이 다른 rgb
label = torch.empty(2, 512, 512, dtype=torch.long)
#%%
for k in mapping:
    idx=(mask==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
    validx = (idx.sum(0) == 3)  # Check that all channels match
    label[mapping.get(k)][validx] = torch.tensor(mapping[k], dtype=torch.long)'''
# %%
# aa=(mask==torch.tensor((0, 0, 0), dtype=torch.uint8))
# # %%
# mapping.get(k)
# %%
# m=PIL2tensor(Image.open('/home/h/Desktop/data/random/test/rgb_label/rgb_label5-2_056.png'))
# print(mask2label(m).shape)
# %%
