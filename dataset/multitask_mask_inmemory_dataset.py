'''
Author: Baoyun Peng
Date: 2022-04-09 12:54:08
LastEditTime: 2022-04-09 13:19:51
Description: 

'''
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

__all__ = ['MultiTaskMaskInMemoryDataset']


class MultiTaskMaskInMemoryDataset (Dataset):
    # --------------------------------------------------------------------------------
    def __init__(self, img_list, meta, mask_list, transform, prefix='data/'):

        self.prefix = prefix
        # read imgs_list and metas
        self.imgs_list = [l.strip() for l in open(img_list).readlines()]
        self.metas = [[int(i) for i in v.strip().split(' ')]
                      for v in open(meta).readlines()[1:]]
        self.mask_list = [l.strip() for l in open(mask_list).readlines()]

        self.transform = transform
        self.img_data_list, self.mask_data_list = self.__readAllData__()

    def __readAllData__(self):
        img_data_list = []
        mask_data_list = []
        for img_path, mask_path in zip(self.imgs_list, self.mask_list):

            img_path = os.path.join(self.prefix, img_path.strip())
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_data_list.append(img)
            mask_path = os.path.join(self.prefix, mask_path.strip())
            mask_data_list.append(cv2.imread(img_path))

        return img_data_list, mask_data_list

    def __getitem__(self, index):
        img  = self.img_data_list[index]
        mask = self.mask_data_list[index]

        concat_img = cv2.merge([img, img, mask])
        if self.transform != None:
            img = self.transform(image=concat_img)['image']
        
        img, _, mask = cv2.split(concat_img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.transpose((2, 0, 1))

        labels = self.metas[index]
        labels = torch.FloatTensor(labels)
        img = torch.FloatTensor(img)
        return img, mask, labels

    def __len__(self):
        return len(self.imgs_list)
