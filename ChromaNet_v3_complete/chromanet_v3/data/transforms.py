"""transforms.py — Data Augmentation"""
from __future__ import annotations
import random
from PIL import Image
import torchvision.transforms.functional as TF


class PairedTransform:
    def __init__(self, image_size: int, augment: bool = True) -> None:
        self.image_size = image_size
        self.augment    = augment

    def __call__(self, img: Image.Image) -> Image.Image:
        img = TF.resize(img, (self.image_size, self.image_size),
                        interpolation=TF.InterpolationMode.BICUBIC)
        if not self.augment:
            return img
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.3:
            w, h = img.size
            scale = random.uniform(0.8, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            left = random.randint(0, w - nw)
            top  = random.randint(0, h - nh)
            img = TF.crop(img, top, left, nh, nw)
            img = TF.resize(img, (h, w), interpolation=TF.InterpolationMode.BICUBIC)
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
            img = TF.adjust_contrast(img,   random.uniform(0.85, 1.15))
        return img


def build_transforms(image_size: int, augment: bool = True) -> PairedTransform:
    return PairedTransform(image_size=image_size, augment=augment)

__all__ = ["PairedTransform", "build_transforms"]
