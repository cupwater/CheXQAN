'''
Author: your name
Date: 2021-10-11 14:45:50
LastEditTime: 2021-10-11 14:45:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CheXQAN/options.py
'''

import argparse

parser = argparse.ArgumentParser(description='Image Classification Training')

# model related, including  Architecture, path, datasets
parser.add_argument('--arch', type=str, default='resnet50', help='network architecture')
parser.add_argument('--num-classes', default=1, type=int,
                    help='the number of classes, default 14')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--prefix', type=str, default='/tmp/znzhang2/medical/', help='image prefix')

# training hyper-parameters
parser.add_argument('--epoch', default=60, type=int, metavar='N',
                    help='train epoches')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
