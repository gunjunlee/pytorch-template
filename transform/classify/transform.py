import torch.nn.functional as F

import cv2
import imgaug as ia
from imgaug import augmenters as iaa

import random


class Transform:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, image):
        wrap = {'image': image}
        config = self.config.get('TRANSFORM')

        if config is None:
            return image

        if config.get('INPUT_SIZE') is not None:
            w, h = config.INPUT_SIZE
            new_size = (w, h)

            wrap['image'] = cv2.resize(wrap['image'], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
        if config.get('ROTATE') is not None:
            if config.ROTATE != 0.0:    
                rotate = config.ROTATE

                wrap['image'] = iaa.Affine(rotate=config.ROTATE).augment_image(wrap['image'])

        # if config.get('FLIP') is not None:
        #     if config

        # if config.get('CROP') is not None:
        #     if len(image.shape) == 3:
        #         orig_height, orig_width, _ = image.shape
        #     else:
        #         orig_height, orig_width = image.shape

        #     scale_min, scale_max = config.CROP
        #     scale = random.uniform(scale_min, scale_max)
        #     w = int(orig_width * scale)
        #     h = int(orig_height * scale)
            
        #     wrap['image'] = cv2.resize(wrap['image'], 
        #                                dsize=(w, h), 
        #                                interpolation=cv2.INTER_NEAREST)
        #     wrap['mask'] = cv2.resize(wrap['mask'], 
        #                               dsize=(w, h), 
        #                               interpolation=cv2.INTER_NEAREST)

        #     if wrap['image'].shape[0] < orig_height:
        #         aug = iaa.PadToFixedSize(height=orig_height,
        #                                  width=orig_width,
        #                                  pad_mode='reflect',
        #                                  position='uniform',
        #                                  deterministic=True)
        #         wrap['image'] = aug.augment_image(wrap['image'])
        #         wrap['mask'] = aug.augment_image(wrap['mask'])
        #     else:
        #         aug = iaa.CropToFixedSize(height=orig_height,
        #                                   width=orig_width,
        #                                   position='uniform',
        #                                   deterministic=True)
        #         wrap['image'] = aug.augment_image(wrap['image'])
        #         wrap['mask'] = aug.augment_image(wrap['mask'])

        return wrap['image']


def get_class_transform(config):
    return Transform(config)
