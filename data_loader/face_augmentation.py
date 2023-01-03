# coding=utf-8

import numpy as np
from albumentations import HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, \
            IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, JpegCompression, \
            DualTransform, RandomResizedCrop

##########################################################
# name:     DataAugmenters
# breif:
#
# usage:
##########################################################
class SquareDataAugmenters:
    def __init__(self, config):
        self.config = config
        self.augmentation =  self.get_aug()

    def _example(self,image,**kwargs):
        """
        example
        :param image: 
        :param kwargs: 
        :return: 
        """
        return image


    def run(self,image,**kwargs):
        """
        augment your image data
        :param image: 
        :param kwargs: 
        :return: 
        """
        data = {'image': image}
        augmented = self.augmentation(**data)
        return augmented['image']

    def get_aug(self, key='pose'):
        if key == 'pose':
            cp = 0.9
            return Compose([
                Flip(),
                OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
                ], p=0.3),
                HueSaturationValue(p=0.3),
                RandomResizedCrop(self.config['img_height'], self.config['img_width'], scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=1, always_apply=False, p=0.7)
            ], p=cp)
    

