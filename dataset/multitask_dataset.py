'''
Author: Baoyun Peng
Date: 2022-02-23 15:42:01
LastEditTime: 2022-04-09 23:32:23
Description: 

'''
import os
import cv2
import torch
from torch.utils.data import Dataset

__all__ = ['MultiTaskDataset']


class MultiTaskDataset (Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, img_list, meta, transform, mask_list=None, prefix='data/'):

        self.prefix = prefix
        # read img_list and metas
        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.metas = [[int(i) for i in v.strip().split(' ')]
                      for v in open(meta).readlines()[1:]]

        if mask_list is not None:
            self.mask_list = [l.strip() for l in open(mask_list).readlines()]
        else:
            self.mask_list = None
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.prefix, self.img_list[index].strip())
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_list is not None:
            mask_path = os.path.join(self.prefix, self.mask_list[index].strip())
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            concat_img = cv2.merge([img, img, mask])
            if self.transform != None:
                img = self.transform(image=concat_img)['image']
                img, _, mask = cv2.split(concat_img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.transpose((2, 0, 1))

        labels = self.metas[index]
        labels = torch.FloatTensor(labels)
        img = torch.FloatTensor(img)
        if self.mask_list is not None:
            mask = torch.FloatTensor(mask)
        else:
            mask = None
        return img, mask, labels


    def __len__(self):
        return len(self.img_list)
