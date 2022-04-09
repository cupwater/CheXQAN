'''
Author: Baoyun Peng
Date: 2022-03-09 16:19:21
LastEditTime: 2022-04-09 23:46:52
Description: 

'''
import os
import cv2
import torch
from torch.utils.data import Dataset

__all__ = ['MultiTaskInMemoryDataset']


class MultiTaskInMemoryDataset (Dataset):
    # --------------------------------------------------------------------------------
    def __init__(self, img_list, meta, transform, mask_list=None, prefix='data/'):

        self.prefix = prefix
        # read imgs_list and metas
        self.imgs_list = [l.strip() for l in open(img_list).readlines()]
        self.metas = [[int(i) for i in v.strip().split(' ')]
                      for v in open(meta).readlines()[1:]]
        
        if mask_list is not None:
            self.mask_list = [l.strip() for l in open(mask_list).readlines()]
        else:
            self.mask_list = None

        self.transform = transform
        self.img_data_list, self.mask_data_list = self.__readAllData__()

    def __readAllData__(self):
        img_data_list = []
        mask_data_list = []
        for index in range(len(self.imgs_list)):
            img_path = os.path.join(self.prefix, self.imgs_list[index].strip())
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_data_list.append(img)
            if self.mask_list is not None:
                mask_path = os.path.join(self.prefix, self.mask_list[index].strip())
                mask_data_list.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        return img_data_list, mask_data_list

    def __getitem__(self, index):
        img = self.img_data_list[index]
        if self.mask_list is not None:
            concat_img = cv2.merge([img, img, self.mask_data_list[index]])
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
            return img, mask, labels
        else:
            return img, labels

    def __len__(self):
        return len(self.imgs_list)
