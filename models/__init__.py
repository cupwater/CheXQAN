from .resnet import *
from .inception_v3 import *
from .inception_v4 import *
from .mobilenet_v2 import *
from .senet import *


def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])
