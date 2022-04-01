'''
Author: Baoyun Peng
Date: 2022-03-10 18:30:59
LastEditTime: 2022-03-13 21:29:15
Description: Since there are some images in Shukang Chest-Xray which the pixel value is totally different from the normal images, We reverse those images by img = 1.0 - img 

'''
import imageio
import sys
import numpy as np
import pdb
import cv2

def main():
    image_prefix  = '/home/znzhang2/datasets/medical/Shukang/ChestXRay-histImages'
    image_prefix_reverse  = '/home/znzhang2/datasets/medical/Shukang/ChestXRay-histImagesReverse'
    mask_prefix  = '/home/znzhang2/datasets/medical/Shukang/ChestXRay-mask'
    mask_prefix_reverse  = '/home/znzhang2/datasets/medical/Shukang/ChestXRay-maskReverse'

    merge_image_prefix = '/home/znzhang2/datasets/medical/Shukang/ChestXRay-mergeImages'
    merge_mask_prefix = '/home/znzhang2/datasets/medical/Shukang/ChestXRay-mergeMask'
    # import matplotlib.pyplot as plt
    imglist = sys.argv[1]
    import os
    with open(imglist, 'r') as fin:
        for line in fin.readlines():
            mask_path = os.path.join(mask_prefix, line.strip())
            mask_path_reverse = os.path.join(mask_prefix_reverse, line.strip())
            mask = imageio.imread(mask_path)
            mask_reverse = imageio.imread(mask_path_reverse)

            mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask_reverse_contours, _ = cv2.findContours(mask_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            img_path = os.path.join(image_prefix, line.strip())
            img_path_reverse = os.path.join(image_prefix_reverse, line.strip())
            # img = imageio.imread(img_path)
            # img_reverse = imageio.imread(img_path_reverse)

            # normal_sum = np.sum(np.multiply(img/255.0, mask/255.0))
            # reverse_sum = np.sum(np.multiply(img_reverse/255.0, mask_reverse/255.0))

            # copy the correct mask and histImages to dst
            # if normal_sum < reverse_sum:
            # we use the number of contours as the standard for judging whether a image is normal 
            if len(mask_contours) < len(mask_reverse_contours):
                os.system(f"cp {mask_path} {merge_mask_prefix}")
                os.system(f"cp {img_path} {merge_image_prefix}")
            else:
                os.system(f"cp {mask_path_reverse} {merge_mask_prefix}")
                os.system(f"cp {img_path_reverse} {merge_image_prefix}")

if __name__ == '__main__':
    main()


