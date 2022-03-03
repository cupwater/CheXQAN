'''
Author: Baoyun Peng
Date: 2022-02-23 15:42:01
LastEditTime: 2022-03-03 23:07:25
Description: 

'''
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class MultiTaskDataset (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, img_list, meta, transform, prefix='data/'):
    
        self.prefix = prefix
        # read imgs_list and metas
        imgs_list = open(img_list).readlines()
        self.imgs_list = [l.strip() for l in imgs_list]
        metas = open(meta).readlines()[1:]
        self.metas = [ [int(i) for i in v.strip().split(' ')]  for v in metas]
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = os.path.join(self.prefix, self.imgs_list[index].strip())
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        labels = self.metas[index]
        labels = torch.FloatTensor(labels)
        if self.transform != None: 
           img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.imgs_list)
