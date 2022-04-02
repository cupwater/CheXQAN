'''
Author: Baoyun Peng
Date: 2022-04-02 09:42:38
LastEditTime: 2022-04-02 10:18:45
Description: 

'''
'''
Author: Baoyun Peng
Date: 2022-02-23 15:42:01
LastEditTime: 2022-03-09 16:23:22
Description: 

'''
import pydicom as dicom
import cv2
import torch
from torch.utils.data import Dataset
from skimage import exposure as ex
import numpy as np
from augmentation.medical_augment import XrayTestTransform

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
    def __init__ (self, new_dicom_list,transform = XrayTestTransform(crop_size=512, img_size=512)):
        self.new_dicom_list = new_dicom_list
        self.transform = transform

    def dcm_tags_scores(self, ds, standard_tags=standard_tags):
        '''
            return the score about tags information completation
            if all the tags of standard_tags are present in dcm, return 100
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
            print('Unable to convert the pixel data, replace with a random image')
            rgb_img = he(cv2.cvtColor(cv2.imread('db/data/test.jpg'), cv2.COLOR_GRAY2RGB))
            return rgb_img, 0

    def __getitem__(self, index):
        # read the dicom
        study_primary_id, file_path = self.new_dicom_list[index]
        ds = dicom.dcmread(file_path)
        tag_score = self.dcm_tags_scores(ds)
        xray_image, state = self.image_from_dicom(ds)
        rgb_img = self.transform(xray_image)['image']
        return rgb_img, tag_score, study_primary_id, state

    def __len__(self):
        return len(self.new_dicom_list)
