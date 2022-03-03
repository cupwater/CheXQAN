#!/bin/bash
###
 # @Author: Baoyun Peng
 # @Date: 2022-01-20 14:03:37
 # @LastEditTime: 2022-03-03 22:13:43
 # @Description: 
 # 
### 

python multitask_main.py \
    --arch densenet121 \
    --lr 0.001 \
    --train-meta selected_train_meta.txt \
    --test-meta selected_test_meta.txt \
    --train-batch 32 \
    --num-classes 13 \
    --prefix data/images \