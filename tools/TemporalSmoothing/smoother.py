"""
Temporal smoothing helpers for colorized frame sequences.

Two modes are supported:

- "legacy_average":
  Sliding-window average with anchor blending. Simple, but can blur motion.
- "flow_chroma":
  Motion-compensated chroma stabilization. Optical flow is computed on the
  grayscale source frames, previous stabilized chroma is warped into the
  current frame, and only Lab `ab` channels are blended. Current luminance is
  always preserved.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _ensure_local_site_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    venv_root = repo_root / ".venv"
    candidates = []
    if os.name == "nt":
        candidates.append(venv_root / "Lib" / "site-packages")
    else:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates.append(venv_root / "lib" / py_ver / "site-packages")
        candidates.append(venv_root / "Lib" / "site-packages")
    for cand in candidates:
        if cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))


_ensure_local_site_packages()

CV2_IMPORT_ERROR = None
try:
    import cv2  # type: ignore
except Exception as exc:
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc

NUMPY_IMPORT_ERROR = None
try:
    import numpy as np  # type: ignore
except Exception as exc:
    np = None  # type: ignore
    NUMPY_IMPORT_ERROR = exc

try:
    from skimage import color as skcolor  # type: ignore
except Exception:
    skcolor = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except ImportError:
    ort = None  # type: ignore


def _collect_frame_names(directory: str) -> list[str]:
    allowed_exts = {".jpg", ".jpeg", ".png"}
    entries = sorted(os.listdir(directory))
    return [name for name in entries if os.path.splitext(name)[1].lower() in allowed_exts]


def _collect_frame_stem_map(directory: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in _collect_frame_names(directory):
        mapping[Path(name).stem] = name
    return mapping


def _prepare_window(frames_dir: str, names: list[str]) -> list["np.ndarray"]:
    window = []
    for fname in names:
        frame = cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_COLOR)
        if frame is not None:
            window.append(frame)
    return window


def _run_numpy_average(frames: list["np.ndarray"]) -> "np.ndarray":
    stacked = np.stack([f.astype(np.float32) for f in frames], axis=0)
    blended = np.clip(np.mean(stacked, axis=0), 0, 255)
    return blended.astype(np.uint8)


def _run_onnx_smooth(session: "ort.InferenceSession", frames: list["np.ndarray"]) -> "np.ndarray":
    input_name = session.get_inputs()[0].name
    base = np.stack([f.astype(np.float32) / 255.0 for f in frames], axis=0)

    candidates = [
        base,
        base[np.newaxis, ...],
        base.transpose(0, 3, 1, 2),
        base.transpose(0, 3, 1, 2)[np.newaxis, ...],
    ]

    last_exc: Exception | None = None
    for cand in candidates:
        try:
            outputs = session.run(None, {input_name: cand})
            result = outputs[0]
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"ONNX inference failed: {last_exc}")

    img = result
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 4 and img.shape[1] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        max_val = float(np.max(img))
        if max_val <= 1.01:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _blend_with_anchor(
    blended: "np.ndarray",
    frames: list["np.ndarray"],
    anchor_weight: float = 0.65,
) -> "np.ndarray":
    if not frames:
        return blended
    anchor = frames[len(frames) // 2]
    if anchor is None:
        return blended

    try:
        if anchor.shape[:2] != blended.shape[:2]:
            anchor = cv2.resize(anchor, (blended.shape[1], blended.shape[0]))
    except Exception:
        return blended

    weight = max(0.0, min(anchor_weight, 1.0))
    if weight <= 0.0:
        return blended

    return cv2.addWeighted(anchor, weight, blended, 1.0 - weight, 0.0)


def _validate_common() -> bool:
    if cv2 is None:
        print("[error] OpenCV (cv2) is not available; temporal smoothing skipped.")
        if CV2_IMPORT_ERROR:
            print(f"[detail] OpenCV import error: {CV2_IMPORT_ERROR}")
        return False
    if np is None:
        print("[error] NumPy is not available; temporal smoothing skipped.")
        if NUMPY_IMPORT_ERROR:
            print(f"[detail] NumPy import error: {NUMPY_IMPORT_ERROR}")
        return False
    return True


def _apply_legacy_average(
    input_folder: str,
    output_folder: str,
    use_onnx: bool,
    window_size: int,
    anchor_weight: float,
) -> None:
    in_dir = os.path.abspath(input_folder)
    out_dir = os.path.abspath(output_folder)
    if in_dir == out_dir:
        raise ValueError("input_folder and output_folder must be different paths.")

    os.makedirs(out_dir, exist_ok=True)
    files = _collect_frame_names(in_dir)
    if not files:
        print("[warn] no image frames found for temporal smoothing.")
        return

    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size -= 1
        if window_size < 3:
            window_size = 3
    half_window = window_size // 2
    total = len(files)

    session = None
    provider_used = "numpy"
    model_path = os.path.join(os.path.dirname(__file__), "temporal_smooth.onnx")

    if use_onnx:
        if ort is None:
            print("[warn] onnxruntime not available. Falling back to NumPy averaging.")
        elif not os.path.exists(model_path):
            print(f"[warn] ONNX model not found at {model_path}. Falling back to NumPy averaging.")
        else:
            priority = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            providers = [p for p in priority if p in available] or ["CPUExecutionProvider"]
            try:
                session = ort.InferenceSession(model_path, providers=providers)
                provider_used = session.get_providers()[0]
                print(f"[info] ONNX smoothing enabled with provider: {provider_used}")
            except Exception as exc:
                session = None
                provider_used = "numpy"
                print(f"[warn] Failed to initialise ONNX session ({exc}). Using NumPy averaging.")

    t0 = time.time()
    for idx, name in enumerate(files):
        start = max(0, idx - half_window)
        end = min(total, idx + half_window + 1)
        window_names = files[start:end]
        frames = _prepare_window(in_dir, window_names)
        if not frames:
            continue

        if session is not None:
            try:
                blended = _run_onnx_smooth(session, frames)
            except Exception as exc:
                print(f"\n[warn] ONNX inference error ({exc}); reverting to NumPy.")
                session = None
                provider_used = "numpy"
                blended = _run_numpy_average(frames)
        else:
            blended = _run_numpy_average(frames)

        blended = _blend_with_anchor(blended, frames, anchor_weight=anchor_weight)
        cv2.imwrite(os.path.join(out_dir, name), blended)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"\r[Temporal Smoothing legacy] {idx + 1}/{total} frames", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n[done] Temporal smoothing complete. provider={provider_used} time={elapsed:.2f}s")


def _load_gray_l_channel(gray_path: str) -> "np.ndarray":
    gray_bgr = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    if gray_bgr is None:
        raise FileNotFoundError(f"Gray frame not found: {gray_path}")
    gray_float = gray_bgr.astype(np.float32) / 255.0
    rgb = np.repeat(gray_float[:, :, None], 3, axis=2)
    return skcolor.rgb2lab(rgb)[:, :, 0]


def _bgr_to_lab(bgr: "np.ndarray") -> "np.ndarray":
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return skcolor.rgb2lab(rgb).astype(np.float32)


def _lab_to_bgr(lab: "np.ndarray") -> "np.ndarray":
    rgb = np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)
    return cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _warp_prev_to_current(prev: "np.ndarray", flow_backward: "np.ndarray") -> "np.ndarray":
    h, w = flow_backward.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_backward[:, :, 0]
    map_y = grid_y + flow_backward[:, :, 1]
    return cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _flow_confidence(curr_gray: "np.ndarray", warped_prev_gray: "np.ndarray") -> "np.ndarray":
    error = np.abs(curr_gray.astype(np.float32) - warped_prev_gray.astype(np.float32))
    confidence = 1.0 - np.clip(error / 24.0, 0.0, 1.0)
    return confidence[:, :, None]


def _apply_flow_chroma(
    gray_input_folder: str,
    color_input_folder: str,
    output_folder: str,
    flow_mix: float,
    motion_strength: float,
) -> None:
    gray_dir = os.path.abspath(gray_input_folder)
    color_dir = os.path.abspath(color_input_folder)
    out_dir = os.path.abspath(output_folder)
    os.makedirs(out_dir, exist_ok=True)

    gray_map = _collect_frame_stem_map(gray_dir)
    color_map = _collect_frame_stem_map(color_dir)
    if not gray_map or not color_map:
        print("[warn] no image frames found for motion-compensated smoothing.")
        return
    common_stems = sorted(set(gray_map) & set(color_map))
    if not common_stems:
        raise ValueError("Gray and color frame sequences do not share frame stems.")

    total = len(common_stems)
    t0 = time.time()

    first_gray_name = gray_map[common_stems[0]]
    first_color_name = color_map[common_stems[0]]
    prev_gray = cv2.imread(os.path.join(gray_dir, first_gray_name), cv2.IMREAD_GRAYSCALE)
    first_color = cv2.imread(os.path.join(color_dir, first_color_name), cv2.IMREAD_COLOR)
    if prev_gray is None or first_color is None:
        raise RuntimeError("Failed to load first frame for motion-compensated smoothing.")

    prev_stable_lab = _bgr_to_lab(first_color)
    prev_stable_gray = prev_gray
    prev_stable_lab[:, :, 0] = _load_gray_l_channel(os.path.join(gray_dir, first_gray_name))
    cv2.imwrite(os.path.join(out_dir, first_color_name), _lab_to_bgr(prev_stable_lab))

    for idx in range(1, total):
        stem = common_stems[idx]
        gray_name = gray_map[stem]
        color_name = color_map[stem]
        curr_gray = cv2.imread(os.path.join(gray_dir, gray_name), cv2.IMREAD_GRAYSCALE)
        curr_color = cv2.imread(os.path.join(color_dir, color_name), cv2.IMREAD_COLOR)
        if curr_gray is None or curr_color is None:
            continue

        curr_lab = _bgr_to_lab(curr_color)
        curr_lab[:, :, 0] = _load_gray_l_channel(os.path.join(gray_dir, gray_name))

        flow_backward = cv2.calcOpticalFlowFarneback(
            curr_gray,
            prev_stable_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=25,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

        warped_prev_ab = _warp_prev_to_current(prev_stable_lab[:, :, 1:3], flow_backward)
        warped_prev_gray = _warp_prev_to_current(prev_stable_gray, flow_backward)
        confidence = _flow_confidence(curr_gray, warped_prev_gray)
        prev_weight = np.clip(flow_mix * confidence * motion_strength, 0.0, 0.92)

        stabilized_lab = curr_lab.copy()
        stabilized_lab[:, :, 1:3] = (
            curr_lab[:, :, 1:3] * (1.0 - prev_weight)
            + warped_prev_ab * prev_weight
        )

        stabilized_bgr = _lab_to_bgr(stabilized_lab)
        cv2.imwrite(os.path.join(out_dir, color_name), stabilized_bgr)

        prev_stable_lab = stabilized_lab
        prev_stable_gray = curr_gray

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(
                f"\r[Temporal Smoothing flow_chroma] {idx + 1}/{total} frames",
                end="",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n[done] Motion-compensated chroma stabilization complete. time={elapsed:.2f}s")


def apply_temporal_smoothing(
    input_folder: str,
    output_folder: str,
    use_onnx: bool = False,
    window_size: int = 9,
    anchor_weight: float = 0.65,
    mode: str = "legacy_average",
    gray_input_folder: str | None = None,
    flow_mix: float = 0.75,
    motion_strength: float = 1.0,
) -> None:
    if not _validate_common():
        return

    selected_mode = (mode or "legacy_average").strip().lower()
    if selected_mode == "legacy_average":
        _apply_legacy_average(
            input_folder=input_folder,
            output_folder=output_folder,
            use_onnx=use_onnx,
            window_size=window_size,
            anchor_weight=anchor_weight,
        )
        return

    if selected_mode == "flow_chroma":
        if skcolor is None:
            print("[error] scikit-image is required for flow_chroma smoothing.")
            return
        if not gray_input_folder:
            raise ValueError("gray_input_folder is required for flow_chroma smoothing.")
        _apply_flow_chroma(
            gray_input_folder=gray_input_folder,
            color_input_folder=input_folder,
            output_folder=output_folder,
            flow_mix=flow_mix,
            motion_strength=motion_strength,
        )
        return

    raise ValueError(f"Unknown temporal smoothing mode: {mode}")
