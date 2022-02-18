'''
Author: your name
Date: 2022-01-26 13:57:27
LastEditTime: 2022-01-26 14:30:17
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /CheXQAN/demo.py
'''
# encoding: utf8

import cv2
import numpy as np
import random
import torch
import json
import models
import pdb
from get_predict import get_predict

from options import parser
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

windows_width = 1200
windows_height = 700
item_w = 120
item_h = 60
numInRow = 4
semantics_list = ['t'+str(i) for i in list(range(20))]

def display(ground_scores, predict_scores, image):
    background = np.ones((windows_height, windows_width, 3))
    resized_image = cv2.resize(image, (windows_height, windows_height))
    background[:, (windows_width-windows_height):windows_width] = resized_image / 255
    # draw the ground-truth scores on image
    cv2.putText(background, '专家标注结果',
                    (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    for idx, _score in enumerate(ground_scores):
        row_idx = 2+int(idx / numInRow)
        col_idx = idx % numInRow
        pos = (item_w*col_idx, item_h*row_idx)
        # pos = (item_h*row_idx, item_w*col_idx)
        cv2.putText(background, semantics_list[idx] + ':' + str(round(_score, 2)),
                    pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    cv2.putText(background, '模型预测标注结果',
                    (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    # draw the predicting scores on image
    for idx, _score in enumerate(predict_scores):
        row_idx = 8 + int(idx / numInRow)
        col_idx = idx % numInRow
        pos = (item_w*col_idx, item_h*row_idx)
        cv2.putText(background, semantics_list[idx] + ':' + str(round(_score, 2)),
                    pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('demo', background)
    key = cv2.waitKey(-1)
    if key == 27:
        exit()

if __name__ == "__main__":

    # Model intialize
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    testdata_json = 'testdata/content.json'
    with open(testdata_json, 'r') as fin:
        test_dict = json.load(fin)
        for key, val in test_dict.items():
            img = cv2.imread(key)
            ground_scores = val['content_score']
            ground_scores = [1 if v=='5' else 0 for v in ground_scores]
        # using model to predict the scores
        predict_scores = get_predict(model, img)
        display(ground_scores, predict_scores, img)