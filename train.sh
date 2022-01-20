#!/bin/bash

python multitask_main.py --arch densenet201 --lr 0.001 --train-batch 32 --epoch 60 --schedule 25 40 --gamma 0.1 --wd 5e-3 --num-classes 19 --prefix data/images