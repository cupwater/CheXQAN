import os
from PIL import Image
import torch
from torch.utils.data import Dataset

import json
#-------------------------------------------------------------------------------- 

class MultiTaskDataset (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, meta_file, transform, prefix='data/'):
    
        self.prefix = prefix
        with open(meta_file) as fin:
            self.meta = json.load(fin)
            self.imglist = list(self.meta.keys())

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
        return len(self.imglist)
