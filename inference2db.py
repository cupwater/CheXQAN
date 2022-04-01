'''
Author: Baoyun Peng
Date: 2022-03-21 22:24:38
LastEditTime: 2022-04-01 14:40:30
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
use_cuda = False


levels = ['D', 'D', 'D', 'C', 'B', 'A' ]


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
    def he(img):
        if(len(img.shape) == 2):  # gray
            outImg = ex.equalize_hist(img[:, :])*255
        elif(len(img.shape) == 3):  # RGB
            outImg = np.zeros((img.shape[0], img.shape[1], 3))
            for channel in range(img.shape[2]):
                outImg[:, :, channel] = ex.equalize_hist(
                    img[:, :, channel])*255

        outImg[outImg > 255] = 255
        outImg[outImg < 0] = 0
        return outImg.astype(np.uint8)
    try:
        # ds = pydicom.dcmread(dicom_files, force=True)
        # convert datatype to float
        new_image = ds.pixel_array.astype(float)
        # Not sure if there are negative values for grayscale in the dicom code
        dcm_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
        dcm_image = dcm_image.astype(np.uint8)
        rgb_img = he(cv2.cvtColor(dcm_image, cv2.COLOR_GRAY2RGB))
        return rgb_img
    except Exception:
        print('Unable to convert the pixel data: one of Pixel Data, Float Pixel Data or Double Float Pixel Data must be present in the dataset')
        return None


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
        elif file_path.strip().split('.')[-1] == 'dcm' and os.path.getmtime(file_path) > last_time:
            study_primary_id = file_path.split('/')[6]
            new_dicom_list.append((study_primary_id, file_path))
    return new_dicom_list


def get_xray_scores(model, data, transform=None, tasks_num=20):
    '''
        using ai_quality_model to evaluate the x-ray images, and return detail scores
    '''
    # transform the data
    model.eval()
    xray_scores = []
    for img in data:
        if img is None:
            xray_scores.append(-1*np.ones(tasks_num))
            continue
        img = transform(image=img)['image']
        img = img.transpose((2, 0, 1))
        input_data = torch.FloatTensor(img)
        input_data = input_data.unsqueeze(0)
        if use_cuda:
            input_data = input_data.cuda()
        predict_score = model(input_data)
        if use_cuda:
            predict_score = predict_score.cpu()
        predict_score = predict_score.data.numpy().reshape(-1)
        predict_score[predict_score>0.5] = 1
        predict_score[predict_score<=0.5] = 0
        xray_scores.append(predict_score)
    return np.array(xray_scores)


def init_ai_quality_model(args):
    '''
        initial the ai quality model
    '''
    global use_cuda
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
    transform = XrayTestTransform(crop_size=512, img_size=512)
    # inference to get the xray_score
    xray_scores = get_xray_scores(model, data, transform)
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
        success_score = score.copy()
        success_score[success_score==-1] = 0
        ai_score = round(1.0*np.sum(success_score) / len(success_score) * 100, 2)
        ai_score_level = levels[int(ai_score/20)]
        state = '2' if np.sum(success_score) < np.sum(score) else '4'
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
    # acquire all the new data and get to image
    new_dicom_list = acquire_incremental_list(last_time)
    if len(new_dicom_list) < 1:
        print('no new data need to inference')
        return
    model = init_ai_quality_model(args)
    study_primary_id_list, scores = inference(model, new_dicom_list)
    update_ai_model_data_center(conn, cursor, study_primary_id_list, scores)
    insert_ai_model_finish_template_info(
        conn, cursor, study_primary_id_list, scores.tolist())


if __name__ == '__main__':
    main()
