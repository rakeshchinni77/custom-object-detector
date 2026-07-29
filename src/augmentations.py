"""Albumentations-based augmentation pipelines for object detection.

This module exposes reusable train/validation transform pipelines that work with
bounding boxes and class labels.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Create the training augmentation pipeline.

    Args:
        image_size: Target image size used by the pipeline.

    Returns:
        An Albumentations Compose object configured for object detection.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


def get_valid_transforms(image_size: int = 640) -> A.Compose:
    """Create the validation augmentation pipeline.

    Args:
        image_size: Target image size used by the pipeline.

    Returns:
        An Albumentations Compose object configured for object detection.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
