"""
Standalone helper to run temporal smoothing on a folder of colorized frames.

This mirrors the Jupyter notebook cell but is easier to execute directly:
    python temporal_smoothing_runner.py
and then follow the prompts for input folder, output folder (auto-generated),
window size, and ONNX usage.
"""

from __future__ import annotations

from pathlib import Path

from tools.TemporalSmoothing import apply_temporal_smoothing


def _normalize_path(path_text: str) -> Path | None:
    """Clean and resolve the user-supplied path string."""
    if not path_text:
        return None
    cleaned = path_text.strip().strip("\"'").replace("\\", "/")
    return Path(cleaned).expanduser().resolve()


def _prompt_input_folder() -> Path:
    while True:
        raw = input("Enter input folder for smoothing: ").strip()
        folder = _normalize_path(raw)
        if folder and folder.exists() and folder.is_dir():
            return folder
        print("[warn] Folder not found. Please provide a valid directory path.")


def _prompt_window_size(default: int = 9) -> int:
    raw = input(f"Enter temporal smoothing window (odd, default={default}): ").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        print(f"[warn] Invalid number; falling back to {default}.")
        value = default
    if value < 3:
        value = 3
    if value % 2 == 0:
        value -= 1
    return max(value, 3)


def _prompt_use_onnx() -> bool:
    raw = input("Use ONNX acceleration if available? [y/N]: ").strip().lower()
    return raw in {"y", "yes"}


def main() -> None:
    input_folder = _prompt_input_folder()
    output_folder = input_folder.parent / f"{input_folder.name}_TemporalSmoothed"
    output_folder.mkdir(parents=True, exist_ok=True)

    window_size = _prompt_window_size()
    use_onnx = _prompt_use_onnx()

    print(
        f"[info] smoothing {input_folder} -> {output_folder} "
        f"(window={window_size}, onnx={use_onnx})"
    )

    apply_temporal_smoothing(
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        use_onnx=use_onnx,
        window_size=window_size,
    )

    print(f"[ok] temporal smoothing complete: {output_folder}")


if __name__ == "__main__":
    main()
