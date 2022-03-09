'''
Author: Baoyun Peng
Date: 2021-10-13 16:36:12
LastEditTime: 2022-03-08 18:28:02
Description: 

'''
from .resnet import *
from .densenet import *
from .densenet_512 import *
from .inception_v3 import *
from .inception_v4 import *
from .mobilenet_v2 import *
from .senet import *


def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])
