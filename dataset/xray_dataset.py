'''
Author: your name
Date: 2021-10-10 14:07:51
LastEditTime: 2021-10-11 14:41:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /chexnet/dataset/DatasetGenerator.py
'''
import os
import cv2
import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class XrayDataset (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, prefix, transform):
    
        self.img_prefix = prefix

        meta_path = os.path.join(prefix, 'meta.txt')
        image_path = os.path.join(prefix, 'image.txt')
        with open(meta_path) as fin:
            lines = fin.readlines()[1:]
            meta_list = [line.strip().split(',') for line in lines]
            self.labels = []
            for meta in meta_list:
                meta = [int(v) for v in meta]
                self.labels.append(meta)
        with open(image_path) as fin:
            lines = fin.readlines()
            self.image_list = [line.strip() for line in lines]
        assert len(self.image_list) == len(self.labels), "image number not match the labels"
        
        self.transform = transform
    
    def __getitem__(self, index):
        img   = cv2.imread(os.path.join(self.prefix, self.image_list[index]))
        label = torch.FloatTensor(self.labels[index])
        if self.transform != None: 
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)
