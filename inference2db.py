'''
Author: Baoyun Peng
Date: 2022-03-21 22:24:38
LastEditTime: 2022-04-06 21:20:49
Description: incremental inference new Chest X-ray images and write the results into database

'''
import torch
import torch.nn as nn

import time
import numpy as np
import os.path
import cv2
from skimage import exposure as ex
import pdb
import pydicom as dicom

import models
from options import parser
from db import table_schema
from db.crud_mysql import *
from augmentation.medical_augment import XrayTestTransform
from dataset import MultiTaskDicomDataset

use_cuda = False
levels = ['D', 'D', 'D', 'C', 'B', 'A' ]
new_dicom_list = []


def acquire_incremental_list(last_time=0, data_path='/data/ks3downloadfromdb/'):
    '''
        acquire the incremental data by create_time
    '''
    global new_dicom_list
    allfilelist = os.listdir(data_path)
    for file in allfilelist:
        file_path = os.path.join(data_path, file)
        if os.path.isdir(file_path):
            acquire_incremental_list(last_time, file_path)
        elif file_path.strip().split('.')[-1] == 'dcm':
            if os.path.getmtime(file_path) > last_time:
                study_primary_id = file_path.split('/')[6]
                new_dicom_list.append((study_primary_id, file_path))


def inference(model):
    '''
        evaluate the dicom by check tags completation and ai_quality_model, return the detail scores
    '''
    global new_dicom_list
    model.eval()

    transform_test = XrayTestTransform(crop_size=512, img_size=512)
    testset = MultiTaskDicomDataset(new_dicom_list, transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=5)

    scores = []
    study_primary_ids = []
    states = []

    for _, (inputs, tag_scores, ids, state) in enumerate(testloader):
        if use_cuda:
            #inputs, tag_scores = inputs.cuda(), tag_scores.cuda()
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)
        xray_scores = model(inputs)
        if use_cuda:
            xray_scores = xray_scores.cpu()

        xray_scores = xray_scores.data.numpy().reshape(xray_scores.size(0), -1)
        xray_scores[xray_scores>0.5] = 1
        xray_scores[xray_scores<=0.5] = 0

        tag_scores = np.array(tag_scores).reshape(tag_scores.size(0), -1)
        _score = np.concatenate((tag_scores, xray_scores), axis=1)
        scores.append( _score)
        study_primary_ids += ids
        states += state
    
    return study_primary_ids, np.array(scores), states


def init_ai_quality_model(args):
    '''
        initial the ai quality model
    '''
    global use_cuda
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, args.num_classes), nn.Sigmoid())
    model.load_state_dict(torch.load(args.pretrained_weights)[
                          'state_dict'], strict=True)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    return model


def update_ai_model_data_center(conn, cursor, study_primary_id_list, scores, states):
    '''
        update the scores of new study_primary_id_list using inference results in database
    '''
    results = []
    for study_primary_id, score, state in zip(study_primary_id_list, scores):
        _condition = f"study_primary_id='{study_primary_id}'"
        success_score = score.copy()
        success_score[success_score==-1] = 0
        ai_score = round(1.0*np.sum(success_score) / len(success_score) * 100, 2)
        ai_score_level = levels[int(ai_score/20)]
        state = '2' if state==1 else '4'
        _new_value = f"ai_score = '{str(ai_score)}', ai_score_level = '{ai_score_level}', state = '{state}'"
        _sql = gen_update_sql('ai_model_data_center', _condition, _new_value)
        results.append(db_execute_val(conn, cursor, _sql))
    return results


def insert_ai_model_finish_template_info(conn, cursor, study_primary_id_list, scores_list):
    '''
        write the detail scores of each dcm into ai_model_finish_template_info
    '''
    # first, select the module information from ai_model_finish_module_info table
    module_info_sql = gen_select_sql('ai_model_template_module_info')
    module_info = db_execute_val(conn, cursor, module_info_sql)

    insert_sql_prefix = gen_insert_sql(
        'ai_model_finish_template_info', table_schema.ai_model_finish_template_info)

    for study_primary_id, scores in zip(study_primary_id_list, scores_list):
        _condition = f"study_primary_id='{study_primary_id}'"
        _sql = gen_select_sql('ai_model_data_center', _condition)
        result = db_execute_val(conn, cursor, _sql)[0]
        val_prefix = tuple(result[1:5] + result[6:7])
        insert_vals = []
        # since 'task_id',
        # 'model_unique_code',
        # 'system_source',
        # 'hospital_code',
        # 'data_time',
        # 'study_primary_id',
        # how to generate id?
        for template_meta, template_score in zip(module_info, scores):
            template_content = '是' if template_score > 0.5 else '否'
            template_score = '1' if template_score > 0.5 else '0'
            val = val_prefix + \
                tuple(template_meta[1:4]) + tuple([template_content, template_score])
            insert_vals.append(val)
        row_count = db_execute_val(
            conn, cursor, insert_sql_prefix, insert_vals)
        print(f"insert {row_count} row number")
    return


def main():
    args = parser.parse_args()
    # init the database connection
    conn, cursor = get_connect(
        'download', 'Down@0221', 'ai_model_quality_control')
    _sql = f"select update_time from information_schema.tables where table_name='ai_model_finish_template_info';"
    result = db_execute_val(conn, cursor, _sql)
    last_time = time.mktime(time.strptime(str(result[0][0]), '%Y-%m-%d %H:%M:%S'))
    last_time = 0
    # acquire all the new data and get to image
    acquire_incremental_list(last_time)
    if len(new_dicom_list) < 1:
        print('no new data need to inference')
        return
    model = init_ai_quality_model(args)
    study_primary_ids, scores, states = inference(model)
    pdb.set_trace()
    update_ai_model_data_center(conn, cursor, study_primary_ids, scores, states)
    insert_ai_model_finish_template_info(
        conn, cursor, study_primary_ids, scores.tolist())


if __name__ == '__main__':
    main()
