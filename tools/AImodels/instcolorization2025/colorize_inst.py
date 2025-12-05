import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import networks
from .siggraph_loader import load_eccv16, load_siggraph17
from . import util as inst_util


def parse_args():
    parser = argparse.ArgumentParser(description="Image colorization (SIGGRAPH17 / ECCV16)")
    parser.add_argument("input", type=str, help="Path to a grayscale image or a folder of images")
    parser.add_argument("--output", type=str, default="results", help="Output folder for colorized images")
    parser.add_argument("--image-size", type=int, default=256, help="Resize images to this square resolution for inference")
    parser.add_argument("--style", type=str, default="siggraph17", choices=["siggraph17", "eccv16"], help="Colorization style")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda|cpu|directml (auto if omitted)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], help="Inference dtype")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to weights (otherwise auto-download)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    return parser.parse_args()


def select_device(requested: Optional[str]) -> torch.device:
    if requested:
        if requested.lower() == "directml":
            try:
                import torch_directml
                return torch_directml.device()
            except Exception:
                print("DirectML requested but torch-directml is not available, falling back to CPU.")
                return torch.device("cpu")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def make_opt(image_size: int) -> SimpleNamespace:
    # minimal option namespace for inst_util/networks
    return SimpleNamespace(
        # color/lab config
        ab_norm=110.0,
        ab_max=110.0,
        ab_quant=10.0,
        l_norm=100.0,
        l_cent=50.0,
        mask_cent=0.5,
        sample_p=1.0,
        sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        # model config
        input_nc=1,
        output_nc=2,
        ngf=64,
        norm="batch",
        no_dropout=False,
        init_type="xavier",
        which_direction="AtoB",
        classification=False,
        fineSize=image_size,
    )


def load_weights(model: torch.nn.Module, path: str, device: torch.device):
    if not Path(path).is_file():
        print(f"[warn] checkpoint not found: {path}")
        return
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # handle DataParallel prefixes
    cleaned = {}
    model_state = model.state_dict()
    for k, v in state.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        if key not in model_state:
            continue
        target_shape = model_state[key].shape
        if v.shape == target_shape:
            cleaned[key] = v
        elif v.ndim == 4 and v.shape[0] == target_shape[0] and v.shape[1] == 1 and target_shape[1] > 1:
            # Adapt first conv weights (e.g., pretrained 1-channel to multi-channel input)
            cleaned[key] = v.repeat(1, target_shape[1], 1, 1) / float(target_shape[1])
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys in {path}: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys in {path}: {unexpected}")


def prepare_transforms(size: int):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.iterdir() if p.suffix.lower() in exts]


def to_device(t: torch.Tensor, device: torch.device, dtype: torch.dtype):
    return t.to(device=device, dtype=dtype)


def run_siggraph(net: torch.nn.Module, img_lab: dict, opt, device, dtype):
    A = to_device(img_lab["A"], device, dtype)
    hint_B = to_device(img_lab["hint_B"], device, dtype)
    mask_B = to_device(img_lab["mask_B"], device, dtype)
    _, out_reg = net(A, hint_B, mask_B)
    return out_reg


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if dtype == torch.float16 and device.type == "cpu":
        print("[warn] float16 on CPU is unsupported; falling back to float32.")
        dtype = torch.float32

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)

    opt = make_opt(args.image_size)
    transform = prepare_transforms(args.image_size)

    # model
    net_siggraph = networks.SIGGRAPHGenerator(
        opt.input_nc + opt.output_nc + 1,
        opt.output_nc,
        norm_layer=networks.get_norm_layer(opt.norm),
        use_tanh=True,
        classification=opt.classification,
    )
    net_siggraph.to(device=device, dtype=dtype)
    net_siggraph.eval()

    if args.weights and Path(args.weights).is_file():
        weight_path = Path(args.weights)
    else:
        weight_path = load_siggraph17() if args.style == "siggraph17" else load_eccv16()
    load_weights(net_siggraph, str(weight_path), device)

    input_root = Path(args.input)
    images = find_images(input_root)
    if not images:
        raise SystemExit(f"No images found at {input_root}")

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        full_tensor = transform(img)
        full_lab = inst_util.get_colorization_data([full_tensor.unsqueeze(0)], opt, ab_thresh=0, p=1.0)

        with torch.inference_mode():
            out_reg = run_siggraph(net_siggraph, full_lab, opt, device, dtype)
            out_rgb = inst_util.lab2rgb(torch.cat((full_lab["A"].to(device, dtype), out_reg), dim=1), opt)

        out_np = torch.clamp(out_rgb, 0.0, 1.0).cpu().numpy()[0].transpose(1, 2, 0)
        out_img = Image.fromarray((out_np * 255).astype(np.uint8))
        out_file = out_dir / f"{img_path.stem}.png"
        out_img.save(out_file)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
