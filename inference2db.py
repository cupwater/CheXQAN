'''
Author: Baoyun Peng
Date: 2022-03-21 22:24:38
LastEditTime: 2022-03-22 00:26:30
Description: incremental inference new Chest X-ray images and write the results into database

'''
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import os.path, time
import pydicom as dicom

import models
from options import parser
from db import crud_mysql
from db.crud_mysql import gen_update_sql, gen_select_sql, gen_insert_sql, get_connect, close_connect, db_execute_val

import matplotlib.pyplot as plt
import PIL.Image as Image

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
    info = {}
    scores = [ 1 if tag in info else 0 for tag in standard_tags]
    return scores

# read dicom file and convert to image
def image_from_dicom(ds):
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

# acquire the incremental data by create_time
def acquire_incremental_list(late_time=0, data_path = '/data/ks3downloadfromdb/'):
    new_dicom_list = []
    for main_dir, dirs, file_name_list in os.walk(data_path):
        for file in file_name_list and len(main_dir.split('/'))>4:
            if os.path.splitext(file)[-1] == 'dcm' and time.ctime(os.path.getmtime()) > late_time:
                file_path = os.path.join(main_dir,file)
                study_primary_id = main_dir.split('/')[3]
                new_dicom_list.append((study_primary_id,  file_path))
    return new_dicom_list

def xray_scores(model, data, transform=None):
    # transform the data
    model.eval()
    return np.ones(len(data))

def init_model(args):
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, args.num_classes), nn.Sigmoid())
    model.load_state_dict(torch.load(args.pretrained_weights)['state_dict'], strict=False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    return model


def inference(model, new_dicom_list):
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
    xray_scores = xray_scores(model, data)
    tags_scores = np.array(tag_scores, dtype=np.float32)
    scores = np.concatenate((xray_scores, tags_scores), axis=1)
    return study_primary_id_list, scores

def update_ai_model_data_center(conn, cursor, study_primary_id_list, scores):
    for study_primary_id, score in zip(study_primary_id_list, scores):
        _condition = f"study_primary_id='{study_primary_id}'"
        ai_score = 1.0*np.sum(score) / len(score) * 100
        _new_value = f"ai_score = {ai_score}"
        _sql = gen_update_sql('ai_model_data_center', _condition, _new_value)
        result = db_execute_val(conn, cursor, _sql)

def main():
    args = parser.parse_args()
    # acquire all the new data and get to image
    new_dicom_list = acquire_incremental_list()
    # init the database connection
    conn, cursor = get_connect('download', 'Down@0221', 'ai_model_quality_control')
    model = init_model(args)
    study_primary_id_list, scores = inference(model, new_dicom_list)
    update_ai_model_data_center(conn, cursor, study_primary_id_list, scores)