'''
Author: Baoyun Peng
Date: 2022-03-21 22:24:38
LastEditTime: 2022-04-08 14:10:30
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
levels = ['D', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'B', 'A', 'A']
new_dicom_list = []
global_mode = 'normal'

# processed_list = []
# failed_list = []
# processed_path = 'logs/succeed_list.txt'
# failed_path = 'logs/failed_list.txt'


# def acquire_processed_list(processed_list, failed_list):
#     '''
#         acquire the processed list and failed list given file path
#     '''
#     # global processed_list, failed_list
#     with open(processed_path) as fin:
#         processed_list = fin.readlines()
#         processed_list = [line.strip() for line in processed_list]

#     failed_list = []
#     with open(failed_path) as fin:
#         failed_list = fin.readlines()
#         failed_list = [line.strip() for line in failed_list]
 

# def acquire_incremental_list(data_path='/data/ks3downloadfromdb/'):
#     '''
#         acquire the incremental data by create_time
#     '''
#     global new_dicom_list
#     allfilelist = os.listdir(data_path)
#     for file in allfilelist:
#         file_path = os.path.join(data_path, file)
#         if os.path.isdir(file_path):
#             acquire_incremental_list(file_path)
#         elif file_path.strip().split('.')[-1] == 'dcm':
#             if file_path not in processed_list and file_path not in failed_list:
#                 study_primary_id = file_path.split('/')[6]
#                 new_dicom_list.append((study_primary_id, file_path))


def db_acquire_incremental_list(conn, cursor, prefix='/data/ks3downloadfromdb/QYZK'):
    '''
        acquire the incremental data by create_time
    '''
    global new_dicom_list
    results = db_execute_val(conn, cursor, 'select * from ai_model_data_center')
    for item in results:
        if int(item[11]) != 2:
            url_paths = item[7].split('?')[0].split('/')
            _posix = os.path.join(url_paths[-4], url_paths[-3], url_paths[-2], url_paths[-1])
            _mid = os.path.join(item[4], item[1], item[6], '000000')
            full_path = os.path.join(prefix, _mid, _posix)
            new_dicom_list.append((item[6], full_path))
            if not os.path.exists(full_path):
                print(full_path)

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


def inference(model, processed_path='logs/succeed.txt', failed_path='logs/failed.txt'):
    '''
        evaluate the dicom by check tags completation and ai_quality_model, return the detail scores
    '''
    global new_dicom_list
    model.eval()

    transform_test = XrayTestTransform(crop_size=512, img_size=512)
    testset = MultiTaskDicomDataset(new_dicom_list, transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=False, num_workers=5)

    scores = []
    study_primary_ids = []
    states = []
    file_paths = []

    for _, (inputs, tag_scores, ids, state, file_path) in enumerate(testloader):
        if use_cuda:
            #inputs, tag_scores = inputs.cuda(), tag_scores.cuda()
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)
        xray_scores = model(inputs)
        if use_cuda:
            xray_scores = xray_scores.cpu()

        xray_scores = xray_scores.data.numpy().reshape(xray_scores.size(0), -1)
        xray_scores[xray_scores > 0.5] = 1
        xray_scores[xray_scores <= 0.5] = 0

        tag_scores = np.array(tag_scores).reshape(tag_scores.size(0), -1)
        _score = np.concatenate((tag_scores, xray_scores), axis=1)
        scores += _score.reshape(-1).tolist()
        study_primary_ids += ids
        states += state
        file_paths += file_path

    failed_list = []
    processed_list = []
    for state, file_path in zip(states, file_paths):
        # state == 0 means failed to parse image, 1 means succeed
        if state == 0:
            failed_list.append(file_path)
        else:
            processed_list.append(file_path)

    scores = np.array(scores).reshape(len(states), -1)
    return study_primary_ids, scores.tolist(), states, processed_list, failed_list


def update_ai_model_data_center(conn, cursor, study_primary_id_list, scores, states, logger_path='logs/execute.txt'):
    '''
        update the scores of new study_primary_id_list using inference results in database
    '''
    logger = open(logger_path, 'w')
    results = []
    for study_primary_id, score, state in zip(study_primary_id_list, scores, states):
        _condition = f"study_primary_id='{study_primary_id}'"
        success_score = score.copy()
        success_score = [ 1 if s >= 0.5 else 0 for s in success_score ]
        ai_score = round(1.0*np.sum(success_score) /
                         len(success_score) * 100, 2)
        ai_score_level = levels[int(ai_score/10)]
        if state == 0:
            ai_score = -1
            _new_value = f"ai_score = '{str(ai_score)}', state = '3'"
        else:
            _new_value = f"ai_score = '{str(ai_score)}', ai_score_level = '{ai_score_level}', state = '2'"
        _sql = gen_update_sql('ai_model_data_center', _condition, _new_value)
        results.append(db_execute_val(conn, cursor, _sql, mode=global_mode, logger=logger))
    logger.close()
    return results


def insert_ai_model_finish_template_info(conn, cursor, study_primary_id_list, scores_list, states, logger_path='logs/execute.txt'):
    '''
        write the detail scores of each dcm into ai_model_finish_template_info
    '''
    logger = open(logger_path, 'a+')
    # first, select the module information from ai_model_finish_module_info table
    module_info_sql = gen_select_sql('ai_model_template_module_info')
    module_info = db_execute_val(conn, cursor, module_info_sql)

    insert_sql_prefix = gen_insert_sql(
        'ai_model_finish_template_info', table_schema.ai_model_finish_template_info)

    for study_primary_id, scores, state in zip(study_primary_id_list, scores_list, states):
        if state == 0:
            continue
        _condition = f"study_primary_id='{study_primary_id}'"
        _sql = gen_select_sql('ai_model_data_center', _condition)
        result = db_execute_val(conn, cursor, _sql)
        if len(result) == 0:
            logger.write(f"error, no such data primary_study_id={study_primary_id} in ai_model_data_center\n")
            continue
        result = result[0]
        val_prefix = tuple(result[1:5] + result[6:7])
        insert_vals = []
        # since 'task_id',
        # 'model_unique_code',
        # 'system_source',
        # 'hospital_code',
        # 'data_time',
        # 'study_primary_id',
        # how to generate id?
        # if state==0:
        #     filter_module_info = module_info[:19]
        #     filter_scores = scores[:19]
        for template_meta, template_score in zip(module_info, scores):
            template_content = '是' if template_score > 0.5 else '否'
            template_score = '1' if template_score > 0.5 else '0'
            val = val_prefix + \
                tuple(template_meta[1:4]) + \
                tuple([template_content, template_score])
            insert_vals.append(val)
        row_count = db_execute_val(
            conn, cursor, insert_sql_prefix, insert_vals, mode=global_mode, logger=logger)
        print(f"insert {row_count} row number")
    logger.close()
    return


def main():
    global processed_path, failed_path

    args = parser.parse_args()
    # init the database connection
    conn, cursor = get_connect(
        'download', 'Down@0221', 'ai_model_quality_control')
    # _sql = f"select update_time from information_schema.tables where table_name='ai_model_finish_template_info';"
    # result = db_execute_val(conn, cursor, _sql)
    # last_time = time.mktime(time.strptime(
    #     str(result[0][0]), '%Y-%m-%d %H:%M:%S'))
    # last_time = 0

    db_acquire_incremental_list(conn, cursor)
    if len(new_dicom_list) < 1:
        print('no new data need to inference')
        return
        
    model = init_ai_quality_model(args)
    study_primary_ids, scores, states, processed_list, failed_list = inference(model)

    # new a logger file according to execution time to record CRUD operations
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    path =os.path.join('logs', time_now) 
    os.makedirs(path)
    with open(os.path.join(path, 'succeed.txt'), 'w') as fout:
        fout.writelines("\n".join(processed_list))
    with open(os.path.join(path, 'failed.txt'), 'w') as fout:
        fout.writelines("\n".join(failed_list))
    
    execute_path = os.path.join(path, 'execute.txt')
    update_ai_model_data_center(
        conn, cursor, study_primary_ids, scores, states, logger_path=execute_path)
    insert_ai_model_finish_template_info(
        conn, cursor, study_primary_ids, scores, states, logger_path=execute_path)


if __name__ == '__main__':
    main()
