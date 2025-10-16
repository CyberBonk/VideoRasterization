from pathlib import Path
from typing import Literal, Optional
from itertools import chain

from tools.AImodels.zhang_model import (
    eccv16, siggraph17, load_img, preprocess_img, postprocess_tens
)

def _pick_model(variant: Literal["eccv16", "siggraph17"], use_gpu: bool):
    if variant == "eccv16":
        model = eccv16(pretrained=True).eval()
    else:
        model = siggraph17(pretrained=True).eval()
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                print("[warn] GPU requested but CUDA not available; using CPU.")
        except Exception:
            print("[warn] torch not available; using CPU.")
    return model

def _save_rgb(img_np, out_path: Path):
    import numpy as np
    from PIL import Image
    arr = (np.clip(img_np, 0.0, 1.0) * 255.0).astype("uint8")
    Image.fromarray(arr).save(str(out_path))

def colorize_dir(
    frames_dir: Path,
    out_dir: Path,
    models_dir: Optional[Path] = None,  # unused for PyTorch variant
    variant: Literal["eccv16", "siggraph17"] = "siggraph17",
    preview: bool = False,
    use_gpu: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _pick_model(variant, use_gpu)

    try:
        import torch
    except Exception:
        torch = None  # type: ignore



    frames_path = Path(frames_dir)
    paths = sorted(
        chain(
            frames_path.glob("*.png"),
            frames_path.glob("*.jpg"),
            frames_path.glob("*.jpeg"),
            frames_path.glob("*.JPG"),
            frames_path.glob("*.JPEG"),
            frames_path.glob("*.PNG"),
        ),
        key=lambda p: p.name,
    )

    if not paths:
        print(f"[error] no PNG frames found in: {frames_path}")
        return

    if preview:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception:
            print("[warn] matplotlib not installed; preview disabled.")
            preview = False

    for p in paths:
        img = load_img(str(p))
        tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

        if use_gpu and torch is not None and torch.cuda.is_available():
            tens_l_rs = tens_l_rs.cuda()

        out_ab = model(tens_l_rs).cpu()
        out_img = postprocess_tens(tens_l_orig, out_ab)
        _save_rgb(out_img, Path(out_dir) / p.name)

        if preview:
            import matplotlib.pyplot as plt
            import numpy as np
            img_bw = postprocess_tens(
                tens_l_orig,
                np.concatenate([0*tens_l_orig.numpy(), 0*tens_l_orig.numpy()], axis=1)
            )
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); plt.imshow(img_bw); plt.title("Input (L)"); plt.axis("off")
            plt.subplot(1,2,2); plt.imshow(out_img); plt.title(f"Output ({variant})"); plt.axis("off")
            plt.tight_layout(); plt.show()
