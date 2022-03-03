#!/bin/bash
###
 # @Author: Baoyun Peng
 # @Date: 2022-01-20 14:03:37
 # @LastEditTime: 2022-03-03 23:12:16
 # @Description: 
 # 
### 

python multitask_main.py \
    --arch densenet121 \
    --lr 0.001 \
    --pretrained-weights checkpoints/pretrained_densenet121.pth.tar \
    --train-list data/train_list.txt \
    --test-list data/test_list.txt \
    --train-meta data/train_meta.txt \
    --test-meta data/test_meta.txt \
    --train-batch 32 \
    --num-classes 13 \
    --prefix data/ \