'''
Author: your name
Date: 2021-10-11 17:07:18
LastEditTime: 2021-10-11 17:10:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CheXQAN/data/gen_meta.py
'''
import numpy as np

img_path = 'train/image.txt'

lines = open(img_path).readlines()

meta_list = [ str(len(lines)) ]
for line in lines:
    if 'normal' in line:
        meta_list.append('1')
    else:
        meta_list.append('0')

with open(img_path.replace('image', 'meta'), 'w') as fout:
    fout.writelines("\n".join(meta_list))