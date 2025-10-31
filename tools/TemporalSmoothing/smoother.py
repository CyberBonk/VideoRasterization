"""
Temporal smoothing helpers for colorized frame sequences.

apply_temporal_smoothing(input_folder, output_folder, use_onnx=False, window_size=9)
    - input_folder: location of colorized frames (jpg/png)
    - output_folder: destination for smoothed frames
    - window_size: odd-sized sliding window for temporal blending
  Uses NumPy averaging by default; if ONNX Runtime is available the exported
  temporal_smooth.onnx model can run on CUDA, DirectML (onnxruntime-directml),
  or CPU providers to produce the smoothed output.
"""

import os
import sys
import time
from pathlib import Path


def _ensure_local_site_packages() -> None:
    """Make sure the repo's virtualenv site-packages is visible on sys.path."""
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
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc

NUMPY_IMPORT_ERROR = None
try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    np = None  # type: ignore
    NUMPY_IMPORT_ERROR = exc

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    ort = None  # type: ignore


def _collect_frame_names(directory: str) -> list[str]:
    allowed_exts = {".jpg", ".jpeg", ".png"}
    entries = sorted(os.listdir(directory))
    return [name for name in entries if os.path.splitext(name)[1].lower() in allowed_exts]


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
        except Exception as exc:  # pragma: no cover - defensive
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


def apply_temporal_smoothing(
    input_folder: str,
    output_folder: str,
    use_onnx: bool = False,
    window_size: int = 9,
) -> None:
    if cv2 is None:
        print("[error] OpenCV (cv2) is not available; temporal smoothing skipped.")
        if CV2_IMPORT_ERROR:
            print(f"[detail] OpenCV import error: {CV2_IMPORT_ERROR}")
        return

    if np is None:
        print("[error] NumPy is not available; temporal smoothing skipped.")
        if NUMPY_IMPORT_ERROR:
            print(f"[detail] NumPy import error: {NUMPY_IMPORT_ERROR}")
        return

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
            except Exception as exc:  # pragma: no cover - defensive
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
            except Exception as exc:  # pragma: no cover - defensive
                print(f"\n[warn] ONNX inference error ({exc}); reverting to NumPy.")
                session = None
                provider_used = "numpy"
                blended = _run_numpy_average(frames)
        else:
            blended = _run_numpy_average(frames)

        cv2.imwrite(os.path.join(out_dir, name), blended)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"\r[Temporal Smoothing] {idx + 1}/{total} frames", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n[done] Temporal smoothing complete. provider={provider_used} time={elapsed:.2f}s")
