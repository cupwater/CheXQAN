'''
Author: Baoyun Peng
Date: 2022-03-09 16:19:21
LastEditTime: 2022-03-09 16:21:36
Description: 

'''
'''
Author: Baoyun Peng
Date: 2022-02-23 15:42:01
LastEditTime: 2022-03-05 13:28:44
Description: 

'''
import os
import cv2
import torch
from torch.utils.data import Dataset

__all__ = ['MultiTaskInMemoryDataset']

class MultiTaskInMemoryDataset (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, img_list, meta, transform, prefix='data/'):
    
        self.prefix = prefix
        # read imgs_list and metas
        imgs_list = open(img_list).readlines()
        self.imgs_list = [l.strip() for l in imgs_list]
        metas = open(meta).readlines()[1:]
        self.metas = [ [int(i) for i in v.strip().split(' ')]  for v in metas]
        self.transform = transform
        self.img_data_lists = self.__readAllData__()
    
    def __readAllData__(self):
        img_data_lists = []
        for index in len(self.imgs_list):
            img_path = os.path.join(self.prefix, self.imgs_list[index].strip())
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_data_lists.append(img)
        return img_data_lists

    def __getitem__(self, index):
        img = self.img_data_lists[index]
        if self.transform != None:
           img = self.transform(image=img)['image']
        img = img.transpose((2,0,1))
        labels = self.metas[index]
        labels = torch.FloatTensor(labels)
        img = torch.FloatTensor(img)
        return img, labels

    def __len__(self):
        return len(self.imgs_list)

