'''
Author: Baoyun Peng
Date: 2022-03-04 23:49:14
LastEditTime: 2022-03-08 09:05:25
Description: 

'''
import albumentations as A

def XrayTrainTransform(img_size=256, crop_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=15),
        A.RandomCrop(width=crop_size, height=crop_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def XrayTestTransform(img_size=256, crop_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CenterCrop(width=crop_size, height=crop_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

if __name__ == "__main__":
    import cv2
    img = cv2.imread('baks/1.jpg')
    transforms = XrayTrainTransform()
    new_img = transforms(image=img)['image']
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('new_img', new_img)
    key=cv2.waitKey(-1)