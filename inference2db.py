'''
Author: Baoyun Peng
Date: 2022-03-21 22:24:38
LastEditTime: 2022-03-22 14:21:03
Description: incremental inference new Chest X-ray images and write the results into database

'''
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import os.path
import time
import pdb
import pydicom as dicom

import models
from options import parser
from db import crud_mysql
from db import table_schema
from db.crud_mysql import *

import matplotlib.pyplot as plt
import PIL.Image as Image

new_dicom_list = []

standard_tags = [
    "StudyID", "PatientName", "PatientSex",
    "PatientBirthDate", "PatientAge", "StudyDate",
    "StudyTime", "InstitutionName", "Manufacturer",
    "ManufacturerModelName", "SoftwareVersions", "KVP",
    "XRayTubeCurrent", "ExposureTime", "DistanceSourceToDetector",
    "StudyDescription", "WindowWidth", "WindowCenter",
    "EntranceDoseInmGy"
]


def dcm_tags_scores(ds, standard_tags=standard_tags):
    '''
        return the score about tags information completation
        if all the tags of standard_tags are present in dcm, return 100
    '''
    dcm_tags = ds.dir()
    scores = [1 if tag in dcm_tags else 0 for tag in standard_tags]
    return scores


def image_from_dicom(ds):
    '''
        read dicom file and convert to image
    '''
    try:
        # ds = pydicom.dcmread(dicom_files, force=True)
        # convert datatype to float
        new_image = ds.pixel_array.astype(float)
        # Not sure if there are negative values for grayscale in the dicom code
        dcm_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
        dcm_image = Image.fromarray(np.uint8(dcm_image))
        return dcm_image
    except AttributeError:
        print('Unable to convert the pixel data: one of Pixel Data, Float Pixel Data or Double Float Pixel Data must be present in the dataset')
        return None


def acquire_incremental_list(late_time=0, data_path='/data/ks3downloadfromdb/'):
    '''
        acquire the incremental data by create_time
    '''
    global new_dicom_list
    allfilelist = os.listdir(data_path)
    for file in allfilelist:
        file_path = os.path.join(data_path, file)
        if os.path.isdir(file_path):
            acquire_incremental_list(late_time, file_path)
        elif file_path.strip().split('.')[-1] == 'dcm' and os.path.getmtime(file_path) > late_time:
            study_primary_id = file_path.split('/')[6]
            new_dicom_list.append((study_primary_id, file_path))
    return new_dicom_list


def get_xray_scores(model, data, transform=None):
    '''
        using ai_quality_model to evaluate the x-ray images, and return detail scores
    '''
    # transform the data
    model.eval()
    xray_scores = []
    for image in data:
        xray_scores.append(np.ones(20))
    return np.array(xray_scores)


def init_ai_quality_model(args):
    '''
        initial the ai quality model
    '''
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, args.num_classes), nn.Sigmoid())
    model.load_state_dict(torch.load(args.pretrained_weights)[
                          'state_dict'], strict=False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    return model


def inference(model, new_dicom_list):
    '''
        evaluate the dicom by check tags completation and ai_quality_model, return the detail scores
    '''
    data = []
    tag_scores = []
    study_primary_id_list = []
    for (study_primary_id, file_path) in new_dicom_list:
        # read the dicom
        ds = dicom.dcmread(file_path)
        tag_score = dcm_tags_scores(ds)
        xray_image = image_from_dicom(ds)
        data.append(xray_image)
        tag_scores.append(tag_score)
        study_primary_id_list.append(study_primary_id)
    # inference to get the xray_score
    xray_scores = get_xray_scores(model, data)
    tags_scores = np.array(tag_scores, dtype=np.float32)
    scores = np.concatenate((tags_scores, xray_scores), axis=1)
    return study_primary_id_list, scores


def update_ai_model_data_center(conn, cursor, study_primary_id_list, scores):
    '''
        update the scores of new study_primary_id_list using inference results in database
    '''
    results = []
    for study_primary_id, score in zip(study_primary_id_list, scores):
        _condition = f"study_primary_id='{study_primary_id}'"
        ai_score = round(1.0*np.sum(score) / len(score) * 100, 2)
        _new_value = f"ai_score = '{str(ai_score)}'"
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

    id = 1
    for study_primary_id, scores in zip(study_primary_id_list, scores_list):
        _condition = f"study_primary_id='{study_primary_id}'"
        _sql = gen_select_sql('ai_model_data_center', _condition)
        result = db_execute_val(conn, cursor, _sql)[0]
        val_prefix = tuple(result[1:7])
        insert_vals = []
        # since 'task_id',
        # 'model_unique_code',
        # 'system_source',
        # 'hospital_code',
        # 'data_time',
        # 'study_primary_id',
        # how to generate id?
        
        for template_meta, template_score in zip(module_info, scores):
            val = tuple([id]) + val_prefix + \
                tuple(template_meta[1:5]) + \
                tuple([template_score]) + tuple(template_meta[6:])
            insert_vals.append(val)
            id += 1
        pdb.set_trace()

        row_count = db_execute_val(conn, cursor, insert_sql_prefix, insert_vals)
        print(f"insert {row_count} row number")
    return


def main():
    args = parser.parse_args()
    # acquire all the new data and get to image
    new_dicom_list = acquire_incremental_list()
    # init the database connection
    conn, cursor = get_connect(
        'download', 'Down@0221', 'ai_model_quality_control')
    model = init_ai_quality_model(args)
    study_primary_id_list, scores = inference(model, new_dicom_list)
    update_ai_model_data_center(conn, cursor, study_primary_id_list, scores)
    insert_ai_model_finish_template_info(
        conn, cursor, study_primary_id_list, scores.tolist())


if __name__ == '__main__':
    main()
