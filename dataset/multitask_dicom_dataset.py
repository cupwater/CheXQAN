'''
Author: Baoyun Peng
Date: 2022-04-02 09:42:38
LastEditTime: 2022-04-09 23:47:06
Description: 

'''
import pydicom as dicom
import cv2
import torch
import os
from torch.utils.data import Dataset
from skimage import exposure as ex
import numpy as np
from augmentation.medical_augment import XrayTestTransform
import pdb

__all__ = ['MultiTaskDicomDataset']

standard_tags = [
    "StudyID", "PatientName", "PatientSex",
    "PatientBirthDate", "PatientAge", "StudyDate",
    "StudyTime", "InstitutionName", "Manufacturer",
    "ManufacturerModelName", "SoftwareVersions", "KVP",
    "XRayTubeCurrent", "ExposureTime", "DistanceSourceToDetector",
    "StudyDescription", "WindowWidth", "WindowCenter",
    "EntranceDoseInmGy"
]

class MultiTaskDicomDataset (Dataset):
    
    #-------------------------------------------------------------------------------- 
    def __init__ (self, dicom_list, transform, mask_list=None, prefix=''):
        self.dicom_list = dicom_list
        self.transform = transform
        self.prefix = prefix
        if mask_list is not None:
            self.mask_list = [l.strip() for l in open(mask_list).readlines()]
        else:
            self.mask_list = None

    def dcm_tags_scores(self, ds, standard_tags=standard_tags):
        '''
            return the score about tags information
        '''
        dcm_tags = ds.dir()
        scores = [1 if tag in dcm_tags else 0 for tag in standard_tags]
        return scores

    def image_from_dicom(self, ds):
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
            return rgb_img, 1
        except Exception:
            img = np.random.rand(512,512,3) * 255
            rgb_img = he(img.astype('uint8'))
            return rgb_img, 0

    def __getitem__(self, index):
        # read the dicom
        study_primary_id, file_path = self.dicom_list[index]
        ds = dicom.dcmread(file_path)
        tag_score = self.dcm_tags_scores(ds)
        xray_image, state = self.image_from_dicom(ds)

        if self.mask_list is not None:
            mask_path = os.path.join(self.prefix, self.mask_list[index].strip())
            mask = cv2.imread(mask_path)
            concat_img = cv2.merge([xray_image, xray_image, mask])
            if self.transform != None:
                img = self.transform(image=concat_img)['image']
                img, _, mask = cv2.split(concat_img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img)

        if self.mask_list is not None:
            mask = torch.FloatTensor(mask)
            return img, np.array(tag_score), study_primary_id, state, file_path, mask
        else:
            return img, np.array(tag_score), study_primary_id, state, file_path

    def __len__(self):
        return len(self.dicom_list)
