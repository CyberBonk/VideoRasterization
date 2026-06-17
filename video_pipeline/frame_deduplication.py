"""Consecutive duplicate-frame middleman for AI colorization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from tools.console import status


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_OUTPUT_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class DuplicateFrameMap:
    original_total: int
    groups: list[list[Path]]
    deduped_dir: Path | None = None

    @property
    def unique_total(self) -> int:
        return len(self.groups)

    @property
    def duplicate_total(self) -> int:
        return self.original_total - self.unique_total

    @property
    def has_duplicates(self) -> bool:
        return self.duplicate_total > 0


def _list_frames(frames_dir: Path) -> list[Path]:
    return sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS],
        key=lambda p: p.name,
    )


def _load_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _same_pixels(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def _available_output_path(out_dir: Path, stem: str) -> Path:
    for ext in _OUTPUT_EXTS:
        candidate = out_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No colorized output found for frame stem: {stem}")


def _new_dedup_dir(frames_dir: Path) -> Path:
    base = frames_dir.parent / f"{frames_dir.name}_deduped"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = frames_dir.parent / f"{frames_dir.name}_deduped_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def deduplicate_consecutive_frames(frames_dir: Path) -> tuple[Path, DuplicateFrameMap]:
    frames_dir = Path(frames_dir)
    frames = _list_frames(frames_dir)
    if not frames:
        return frames_dir, DuplicateFrameMap(original_total=0, groups=[])

    groups: list[list[Path]] = []
    previous_pixels: np.ndarray | None = None
    for frame in frames:
        pixels = _load_pixels(frame)
        if groups and previous_pixels is not None and _same_pixels(previous_pixels, pixels):
            groups[-1].append(frame)
        else:
            groups.append([frame])
        previous_pixels = pixels

    frame_map = DuplicateFrameMap(original_total=len(frames), groups=groups)
    if not frame_map.has_duplicates:
        status(f"[dedupe] no exact consecutive duplicate frames in {len(frames)} frames")
        return frames_dir, frame_map

    deduped_dir = _new_dedup_dir(frames_dir)
    deduped_dir.mkdir(parents=True, exist_ok=False)
    for group in groups:
        shutil.copy2(group[0], deduped_dir / group[0].name)

    frame_map = DuplicateFrameMap(
        original_total=len(frames),
        groups=groups,
        deduped_dir=deduped_dir,
    )
    status(
        f"[dedupe] {frame_map.duplicate_total}/{frame_map.original_total} "
        f"exact consecutive duplicates skipped before AI model"
    )
    return deduped_dir, frame_map


def expand_duplicate_outputs(out_dir: Path, frame_map: DuplicateFrameMap) -> None:
    if not frame_map.has_duplicates:
        return

    out_dir = Path(out_dir)
    restored = 0
    for group in frame_map.groups:
        leader_output = _available_output_path(out_dir, group[0].stem)
        output_ext = leader_output.suffix
        for duplicate_frame in group[1:]:
            shutil.copy2(leader_output, out_dir / f"{duplicate_frame.stem}{output_ext}")
            restored += 1

    status(f"[dedupe] restored {restored} duplicate colorized frames")


__all__ = [
    "DuplicateFrameMap",
    "deduplicate_consecutive_frames",
    "expand_duplicate_outputs",
]
