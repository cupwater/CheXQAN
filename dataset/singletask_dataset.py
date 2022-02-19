'''
Author: your name
Date: 2022-02-18 14:22:07
LastEditTime: 2022-02-19 10:19:40
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /CheXQAN/dataset/simgletask_dataset.py
'''
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class SingleTaskDataset (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, img_list, meta_file, transform, prefix='data/'):
    
        self.prefix = prefix
        # read imgs_list and metas
        imgs_list = open(img_list).readlines()
        self.imgs_list = [l.strip() for l in img_list]
        metas = open(meta_file).readlines()[1:]
        self.metas = [int(v) for v in metas]
        self.transform = transform
    
    def __getitem__(self, index):

        img_path = os.path.join(self.prefix, self.imglist[index].strip() + '.jpg')
        while not os.path.exists(img_path):
            index += 1
            img_path = os.path.join(self.prefix, self.imglist[index].strip() + '.jpg')
        img = Image.open(img_path)
        img = img.resize((256, 256),Image.ANTIALIAS)
        img = img.convert(mode='RGB')

        labels = self.meta[self.imglist[index]]['content_score']
        labels = [1 if v=='5' else 0 for v in labels]
        labels = torch.FloatTensor(labels)
        if self.transform != None: 
           img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.imgs_list)
