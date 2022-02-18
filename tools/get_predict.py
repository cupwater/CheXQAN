'''
Author: your name
Date: 2022-01-26 14:09:04
LastEditTime: 2022-01-26 14:09:05
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /CheXQAN/get_predict.py
'''

import cv2
import numpy as np
import random
import torch
import models


def get_predict(model, image):
    # process the image as trianing period
    inputs = torch.autograd.Variable(torch.randn(1, 3, 224, 224), volatile=True)
    outputs = model(inputs)

    
