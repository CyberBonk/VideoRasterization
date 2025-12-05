from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np


def read_image(path: Union[str, Path]) -> Image.Image:
    """Load an image and ensure RGB output."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert PIL image to RGB if needed."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_numpy_uint8(img: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 numpy array (H, W, 3)."""
    img = ensure_rgb(img)
    return np.asarray(img, dtype=np.uint8)
