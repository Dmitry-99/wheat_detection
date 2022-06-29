import albumentations as A
from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize,Flip,
                            Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2


# Albumentations
train_augmentations = A.Compose([
                                A.Flip(0.5),
                                ToTensorV2(p=1.0)
                                ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

valid_augmentations =  A.Compose([
                                ToTensorV2(p=1.0)
                                ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

test_augmentations = A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])