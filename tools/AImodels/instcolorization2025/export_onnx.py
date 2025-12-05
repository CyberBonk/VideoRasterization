import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import networks
from .siggraph_loader import load_eccv16, load_siggraph17


def make_opt(image_size: int):
    return SimpleNamespace(
        ab_norm=110.0,
        ab_max=110.0,
        ab_quant=10.0,
        l_norm=100.0,
        l_cent=50.0,
        mask_cent=0.5,
        sample_p=1.0,
        sample_Ps=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        input_nc=1,
        output_nc=2,
        ngf=64,
        norm="batch",
        classification=False,
        fineSize=image_size,
    )


def load_weights(model: torch.nn.Module, path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
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
            cleaned[key] = v.repeat(1, target_shape[1], 1, 1) / float(target_shape[1])
    model.load_state_dict(cleaned, strict=False)


def export(model, dummy_inputs, output_path: Path, dynamic: bool):
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input_A": {0: "batch", 2: "height", 3: "width"},
            "input_B": {0: "batch", 2: "height", 3: "width"},
            "mask_B": {0: "batch", 2: "height", 3: "width"},
            "out_reg": {0: "batch", 2: "height", 3: "width"},
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path.as_posix(),
        input_names=["input_A", "input_B", "mask_B"],
        output_names=["out_reg"],
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )
    print(f"Saved ONNX model to {output_path}")


def validate(output_path: Path, height: int, width: int):
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"[warn] onnxruntime not available, skipping validation: {e}")
        return
    providers = ["CPUExecutionProvider"]
    if "DmlExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "DmlExecutionProvider")
    session = ort.InferenceSession(output_path.as_posix(), providers=providers)
    dummy = np.random.rand(1, 1, height, width).astype(np.float32)
    mask = np.zeros_like(dummy)
    hint = np.zeros((1, 2, height, width), dtype=np.float32)
    outputs = session.run(None, {"input_A": dummy, "input_B": hint, "mask_B": mask})
    print(f"ONNXRuntime providers: {providers}; output shape: {outputs[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="Export colorization generator to ONNX")
    parser.add_argument("--style", type=str, default="siggraph17", choices=["siggraph17", "eccv16"], help="Model style to export")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to weights (otherwise auto-download)")
    parser.add_argument("--output", type=str, default="checkpoints/base/siggraph.onnx", help="Destination ONNX file")
    parser.add_argument("--height", type=int, default=256, help="Dummy height for export")
    parser.add_argument("--width", type=int, default=256, help="Dummy width for export")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic spatial dims")
    parser.add_argument("--validate", action="store_true", help="Run ONNXRuntime validation after export")
    args = parser.parse_args()

    device = torch.device("cpu")
    opt = make_opt(max(args.height, args.width))
    net = networks.SIGGRAPHGenerator(
        opt.input_nc + opt.output_nc + 1,
        opt.output_nc,
        norm_layer=networks.get_norm_layer(opt.norm),
        use_tanh=True,
        classification=opt.classification,
    )
    net.to(device)
    net.eval()

    if args.weights and Path(args.weights).is_file():
        weight_path = Path(args.weights)
    else:
        weight_path = load_siggraph17() if args.style == "siggraph17" else load_eccv16()
    load_weights(net, weight_path, device)

    input_A = torch.randn(1, opt.input_nc, args.height, args.width, device=device)
    input_B = torch.zeros(1, opt.output_nc, args.height, args.width, device=device)
    mask_B = torch.zeros(1, 1, args.height, args.width, device=device)

    export(net, (input_A, input_B, mask_B), Path(args.output), args.dynamic)
    if args.validate:
        validate(Path(args.output), args.height, args.width)


if __name__ == "__main__":
    main()
