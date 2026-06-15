"""dataset.py — ColorizationDataset (COCO / any image folder)"""
from __future__ import annotations
import os, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor
from torch.utils.data import DataLoader, Dataset
from .transforms import build_transforms

_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}


def _collect(root: str | Path) -> list[Path]:
    paths: list[Path] = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            if Path(f).suffix.lower() in _EXTS:
                paths.append(Path(dp) / f)
    return sorted(paths)


def _to_lab_tensors(img: Image.Image):
    img  = img.convert("RGB")
    rgb  = np.array(img, dtype=np.float32) / 255.0
    lab  = skcolor.rgb2lab(rgb).astype(np.float32)
    L    = torch.from_numpy((lab[:,:,0] / 50.0) - 1.0).unsqueeze(0)
    AB   = torch.from_numpy(np.stack([lab[:,:,1], lab[:,:,2]], 0) / 110.0).clamp(-1,1)
    RGB  = torch.from_numpy(rgb.transpose(2,0,1))
    return L, AB, RGB


class ColorizationDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256,
                 augment: bool = True, max_samples: int | None = None,
                 paths: list[Path] | None = None) -> None:
        self.paths = list(paths) if paths is not None else _collect(root)
        if not self.paths:
            raise RuntimeError(f"No images found under: {root}")
        if max_samples:
            random.shuffle(self.paths)
            self.paths = self.paths[:max_samples]
        self.transform = build_transforms(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0, len(self)-1))
        img = self.transform(img)
        L, AB, RGB = _to_lab_tensors(img)
        return {"L": L, "AB": AB, "RGB": RGB, "path": str(self.paths[idx])}


def build_dataloaders(cfg: dict):
    dc  = cfg.get("data", {})
    root = dc.get("root", "./datasets/coco")
    paths = _collect(root)
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(paths), generator=generator).tolist()
    nv = max(1, int(len(paths) * dc.get("val_split", 0.05)))
    val_paths = [paths[i] for i in order[:nv]]
    train_paths = [paths[i] for i in order[nv:]]
    size = dc.get("image_size", 256)
    tr = ColorizationDataset(root, size, augment=dc.get("augment", True), paths=train_paths)
    vl = ColorizationDataset(root, size, augment=False, paths=val_paths)
    nt = len(tr)
    workers = dc.get("num_workers", 4)
    kw = dict(
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    bs  = dc.get("batch_size", 32)
    print(f"[data] train={nt} val={nv} batch={bs}")
    return (DataLoader(tr, bs, shuffle=True,  drop_last=True,  **kw),
            DataLoader(vl, bs, shuffle=False, drop_last=False, **kw))

__all__ = ["ColorizationDataset", "build_dataloaders"]
