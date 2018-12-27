import torch.nn.functional as F

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from albumentations import (
    Rotate, RandomScale, VerticalFlip, HorizontalFlip,
    RandomBrightness, RandomContrast, MotionBlur
)

import random


class Transform:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, image, mask):
        wrap = {'image': image, 'mask': mask}
        config = self.config.get('TRANSFORM')

        if config is None:
            return image, mask

        if config.get('INPUT_SIZE') is not None:
            w, h = config.INPUT_SIZE
            new_size = (w, h)

            wrap['image'] = cv2.resize(wrap['image'], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            wrap['mask'] = cv2.resize(wrap['mask'], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
        if config.get('ROTATE') is not None:
            if config.ROTATE != 0.0:    
                rotate = config.ROTATE

                wrap = Rotate(limit=(-rotate, rotate), 
                                    interpolation=cv2.INTER_LINEAR,
                                    p=1.0, 
                                    )(image=wrap['image'], mask=wrap['mask'])
        
        if config.get('CROP') is not None:
            if len(image.shape) == 3:
                orig_height, orig_width, _ = image.shape
            else:
                orig_height, orig_width = image.shape

            scale_min, scale_max = config.CROP
            scale = random.uniform(scale_min, scale_max)
            w = int(orig_width * scale)
            h = int(orig_height * scale)
            
            wrap['image'] = cv2.resize(wrap['image'], 
                                       dsize=(w, h), 
                                       interpolation=cv2.INTER_NEAREST)
            wrap['mask'] = cv2.resize(wrap['mask'], 
                                      dsize=(w, h), 
                                      interpolation=cv2.INTER_NEAREST)

            if wrap['image'].shape[0] < orig_height:
                aug = iaa.PadToFixedSize(height=orig_height,
                                         width=orig_width,
                                         pad_mode='reflect',
                                         position='uniform',
                                         deterministic=True)
                wrap['image'] = aug.augment_image(wrap['image'])
                wrap['mask'] = aug.augment_image(wrap['mask'])
            else:
                aug = iaa.CropToFixedSize(height=orig_height,
                                          width=orig_width,
                                          position='uniform',
                                          deterministic=True)
                wrap['image'] = aug.augment_image(wrap['image'])
                wrap['mask'] = aug.augment_image(wrap['mask'])

        if config.get('BRIGHTNESS') is not None:
            brightness = config.BRIGHTNESS
            if brightness != 0.0:
                wrap = RandomBrightness(limit=brightness)(
                    image=wrap['image'],
                    mask=wrap['mask']
                )
        
        if config.get('CONTRAST') is not None:
            contrast = config.CONTRAST
            if contrast != 0.0:
                wrap = RandomContrast(limit=contrast)(
                    image=wrap['image'],
                    mask=wrap['mask']
                )
        
        if config.get('MOTIONBLUR') is not None:
            motionblur = config.MOTIONBLUR
            if motionblur != 0:
                wrap = MotionBlur(blur_limit=motionblur, p=1.0)(
                    image=wrap['image'],
                    mask=wrap['mask']
                )
        
        if config.get('FLIP') is not None:
            wrap = VerticalFlip(p=config.FLIP.VFLIP)(
                image=wrap['image'],
                mask=wrap['mask']
            )

            wrap = HorizontalFlip(p=config.FLIP.HFLIP)(
                image=wrap['image'],
                mask=wrap['mask']
            )

        return wrap['image'], wrap['mask']


def get_seg_transform(config):
    return Transform(config)
