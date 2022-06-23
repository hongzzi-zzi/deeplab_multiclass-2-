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
def mask2label(mask_ori):## mask_ori: RGB tensor
    mask=tensor2PIL(mask_ori)# PIL
    pix=mask.load()# Pixel
    w, h=mask.size
    label = torch.zeros(2, h, w, dtype=torch.float)# new tensir(CHW)
    for k in mapping:
        v=mapping.get(k)
        # w:가로 h: 세로
        for w, h in itertools.product(range(w),range(h)):
            label[v][h][w]=(pix[w, h]==k)
    return label ## tensor
#%%
mask=Image.open('/home/h/Desktop/data/4-4/t_label/t_label4-4_001.png').convert('RGB')
# %% pillow
mask.mode#'RGB'
mask.size# (3300, 2475) w*h(가로*세로)
px=mask.load()# pixel: px[w, h]
print(px[2000, 1400])# w=2000, h=1400 의 RGB픽셀값(204, 215, 197)
print(px[3000, 0])# w=3000, h=0 의 RGB픽셀값(0, 0, 0)
# print(px[2000, 3000])# IndexError: image index out of range
# %%
mask_t=PIL2tensor(mask)
# %%
mask_t.shape# torch.Size([3, 2475, 3300]) CHW(채널 세로 가로)
label=mask2label(mask_t)
# %%
label.shape#torch.Size([2, 2475, 3300])
#%%
mask=Image.open('/home/h/Desktop/data/random/test/rgb_label/rgb_label1_021.png').convert('RGB')
label=mask2label(PIL2tensor(mask))
label.shape# torch.Size([2, 512, 512])
#%%
tensor2PIL(label).mode# 'LA'
tensor2PIL(label).resize((512, 512)).save('a.png')# 'LA' -> 알파채널 0인애들은 다투명해짐(background) 그리고 안투명한부부ㄴ(teeth)의 L채널값은 1이니까 검정 되는거다
tensor2PIL(label[0]).resize((512, 512)).save('aa.png')## background 가 흰색
tensor2PIL(label[1]).resize((512, 512)).save('aaa.png')## teeth가 흰색
#%%

# %%
def mask2RGBmask(list, path):## 굳이 안써도 되지만,,, 2가지 이상의 채널일 경우도 있으니까 ㅇㅅㅇ
    for i in list:
        mask_ori=Image.open(i).resize((512, 512)).convert('RGBA').split()[-1]
        pix=mask_ori.load()
        w, h=mask_ori.size
        rgb_mask=Image.new(mode="RGB", size=(512, 512),color=(0, 0, 0))
        for w, h in itertools.product(range(w),range(h)):
            if pix[w, h]!=0:
                rgb_mask.load()[w, h]=(255, 255, 255)
        rgb_mask.save(path)
# %%
a=['/home/h/Desktop/data/1/t_label/t_label1_021.png']
mask2RGBmask(a, 'b.png')##잘됨
label.shape
# %%
