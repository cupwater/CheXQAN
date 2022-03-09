'''
Training script for Image Classification 
Copyright (c) Baoyun Peng, 2021
'''
from __future__ import print_function


import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from augmentation.medical_augment import XrayTrainTransform, XrayTestTransform


import models
from dataset.multitask_dataset import MultiTaskDataset
from utils import Logger, AverageMeter, mkdir_p, progress_bar
from options import parser

state = {}
best_acc = 0  # best test accuracy

def main():
    global state
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    args.save_path = 'experiments/' + args.dataset + '/' + args.arch
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    global best_acc
    if not os.path.isdir(args.save_path):
        mkdir_p(args.save_path)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = XrayTrainTransform()
    transform_test = XrayTestTransform()
    
    trainset = MultiTaskDataset(args.train_list, args.train_meta, transform_train, prefix=args.prefix)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=5)
    testset = MultiTaskDataset(args.test_list, args.test_meta, transform_test, prefix=args.prefix)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=5)

    # Model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, args.num_classes), nn.Sigmoid())
    model.load_state_dict(torch.load(args.pretrained_weights)['state_dict'], strict=False)
    model = model.cuda()
    cudnn.benchmark = True

    # optimizer and scheduler
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    # logger
    title = 'Chest X-ray Image Quality Assessment using ' + args.arch
    logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Train and val
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        # scheduler.step(losstensor.data[0])

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epoch, state['lr']))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path=args.save_path)

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

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)

        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        loss = criterion(outputs, targets)
        predict = outputs > 0.5
        predict_res = (predict == targets)
        losses.update(loss.item(), inputs.size(0))
        top1.update(torch.sum(predict_res.long())/(inputs.size(0)*targets.size(1)), predict_res.size(0))
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
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        loss = criterion(outputs, targets)
        predict = outputs > 0.5
        predict_res = (predict == targets)
        losses.update(loss.item(), inputs.size(0))
        
        top1.update(torch.sum(predict_res.long())/(inputs.size(0)*targets.size(1)), predict_res.size(0))

        progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f'
                    % (losses.avg, top1.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, save_path='experiment/template', filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, args):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
