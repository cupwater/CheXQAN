'''
Training script for Image Classification 
Copyright (c) Baoyun Peng, 2021
'''
from __future__ import print_function


import os
import shutil
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from augmentation.medical_augment import XrayTrainTransform, XrayTestTransform


import models
import dataset
from utils import Logger, AverageMeter, mkdir_p, progress_bar
from options import parser

state = {}
best_acc = 0
use_cuda = False


def main(config_file):
    global state, best_acc, use_cuda

    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    state['lr'] = common_config['lr']
    if not os.path.isdir(common_config['save_path']):
        mkdir_p(common_config['save_path'])
    use_cuda = torch.cuda.is_available()

    data_config = config['dataset']
    # Dataset and Dataloader
    transform_train = XrayTrainTransform(
        crop_size=data_config['crop_size'], img_size=data_config['img_size'])
    transform_test = XrayTestTransform(
        crop_size=data_config['crop_size'], img_size=data_config['img_size'])
    print('==> Preparing dataset %s' % data_config['type'])

    # get mask_list from config
    train_mask = data_config['train_mask'] if 'mask' in data_config else None
    test_mask = data_config['test_mask'] if 'mask' in data_config else None

    # create dataset for training and testing
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], data_config['train_meta'], transform_train,
        mask_list=train_mask, prefix=data_config['prefix'])
    testset = dataset.__dict__[data_config['type']](
        data_config['test_list'], data_config['test_meta'], transform_test,
        mask_list=test_mask, prefix=data_config['prefix'])

    # create dataloader for training and testing
    trainloader = data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    testloader = data.DataLoader(
        testset, batch_size=common_config['train_batch'], shuffle=False, num_workers=5)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']](
        num_classes=data_config['num_classes'])
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, data_config['num_classes']), nn.Sigmoid())
    model.load_state_dict(torch.load(common_config['pretrained_weights'])[
                          'state_dict'], strict=False)
    if use_cuda:
        model = model.cuda()
    cudnn.benchmark = True

    # optimizer and scheduler
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        momentum=0.9,
        weight_decay=common_config['weight_decay'])

    # logger
    title = 'Chest X-ray Image Quality Assessment using ' + \
        common_config['arch']
    logger = Logger(os.path.join(
        common_config['save_path'], 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss',
                     'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        train_loss, train_acc = train(
            trainloader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss,
                      test_loss, train_acc, test_acc])
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path=common_config['save_path'])

    logger.close()
    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, mask, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        if mask:
            mask = mask.cuda()
            mask = torch.autograd.Variable(mask)
        
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        loss = criterion(outputs, targets)
        predict = outputs > 0.5
        predict_res = (predict == targets)
        losses.update(loss.item(), inputs.size(0))
        top1.update(torch.sum(predict_res.long()) /
                    (inputs.size(0)*targets.size(1)), predict_res.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2f'
                     % (losses.avg, top1.avg, top5.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)


def test(testloader, model, criterion, use_cuda):
    global best_acc
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        loss = criterion(outputs, targets)
        predict = outputs > 0.5
        predict_res = (predict == targets)
        losses.update(loss.item(), inputs.size(0))
        top1.update(torch.sum(predict_res.long()) /
                    (inputs.size(0)*targets.size(1)), predict_res.size(0))

        progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f'
                     % (losses.avg, top1.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            save_path, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, config):
    global state
    if epoch in config['scheduler']:
        state['lr'] *= config['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Medical Quality Assessment using AI')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str,
                        default='experiments/template/config.yaml')
    args = parser.parse_args()
    main(args.config_file)
