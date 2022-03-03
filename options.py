'''
Author: Baoyun Peng
Date: 2022-01-20 10:43:15
LastEditTime: 2022-03-03 21:58:32
Description: 

'''

import argparse

parser = argparse.ArgumentParser(description='Medical Quality Assessment using AI')

# model related, including  Architecture, path, datasets
parser.add_argument('--arch', type=str, default='densenet121', help='network architecture')
parser.add_argument('--num-classes', default=1, type=int,
                    help='the number of classes, default 1')

parser.add_argument('--dataset', type=str, default='XrayDataset', help='dataset')
parser.add_argument('--prefix', type=str, default='/tmp/znzhang2/medical/', help='image prefix')

parser.add_argument('--train-list', type=str, default='train_list.txt', help='train image list')
parser.add_argument('--train-meta', type=str, default='train_meta.txt', help='train meta list')
parser.add_argument('--test-list', type=str, default='test_list.txt', help='test image list')
parser.add_argument('--test-meta', type=str, default='test_meta.txt', help='test meta list')

# training hyper-parameters
parser.add_argument('--epoch', default=60, type=int, metavar='N',
                    help='train epoches')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25, 40],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
