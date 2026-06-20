"""
ChromaStudio — Desktop Application Host
Bridges the React-style HTML/JS frontend with the VideoRasterization Python backend.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import webview

ROOT = Path(__file__).resolve().parent.parent
GUI_DIR = Path(__file__).resolve().parent
SRC_DIR = GUI_DIR / "src"

# ─────────────────────────────────────────────────────────────────────────────
#  System metrics (psutil optional, graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


def _get_system_metrics() -> dict:
    metrics: dict[str, Any] = {
        "cpu_percent": 0,
        "cpu_per_core": [],
        "ram_used_gb": 0,
        "ram_total_gb": 0,
        "ram_percent": 0,
        "disk_read_mb": 0,
        "disk_write_mb": 0,
        "gpu_name": None,
        "gpu_percent": 0,
        "vram_used_gb": 0,
        "vram_total_gb": 0,
        "has_cuda": False,
    }
    if HAS_PSUTIL:
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        metrics["cpu_per_core"] = psutil.cpu_percent(interval=None, percpu=True)
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"] = round(vm.used / (1024**3), 1)
        metrics["ram_total_gb"] = round(vm.total / (1024**3), 1)
        metrics["ram_percent"] = vm.percent
        try:
            io = psutil.disk_io_counters()
            metrics["disk_read_mb"] = round(io.read_bytes / 1e6, 1) if io else 0
            metrics["disk_write_mb"] = round(io.write_bytes / 1e6, 1) if io else 0
        except Exception:
            pass
    if HAS_TORCH and torch.cuda.is_available():
        metrics["has_cuda"] = True
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.mem_get_info(0)
        total = torch.cuda.get_device_properties(0).total_memory
        metrics["vram_total_gb"] = round(total / (1024**3), 1)
        metrics["vram_used_gb"] = round((total - mem[0]) / (1024**3), 1)
    return metrics


def _get_hardware_summary() -> dict:
    cores = os.cpu_count() or 1
    cpu_name = platform.processor() or "CPU"
    summary = {
        "cpu_name": cpu_name,
        "cpu_cores": cores,
        "has_cuda": False,
        "has_ipex": False,
        "gpu_name": None,
        "vram_gb": None,
        "ram_gb": 0,
        "python_version": sys.version.split()[0],
    }
    if HAS_PSUTIL:
        summary["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    if HAS_TORCH:
        if torch.cuda.is_available():
            summary["has_cuda"] = True
            summary["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            summary["vram_gb"] = round(props.total_memory / (1024**3), 1)
        try:
            import intel_extension_for_pytorch  # noqa: F401
            summary["has_ipex"] = True
        except ImportError:
            pass
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Model discovery
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    {
        "id": "colorize_zhang_siggraph17",
        "name": "Zhang siggraph17",
        "description": "Classic 2017 deep-learning colorizer. Best balance of speed and quality.",
        "architecture": "CNN — LAB colorspace",
        "speed_tier": 3,
        "quality_tier": 3,
        "gpu_required": False,
        "status": "ready",
        "variant": "siggraph17",
        "backend": "colorize_zhang",
    },
    {
        "id": "colorize_zhang_eccv16",
        "name": "Zhang eccv16",
        "description": "2016 variant. Faster, slightly lower saturation.",
        "architecture": "CNN — LAB colorspace",
        "speed_tier": 4,
        "quality_tier": 2,
        "gpu_required": False,
        "status": "ready",
        "variant": "eccv16",
        "backend": "colorize_zhang",
    },
    {
        "id": "instcolorization2025",
        "name": "InstColorization 2025",
        "description": "Instance-aware colorization. Better object boundary handling.",
        "architecture": "Instance-aware CNN",
        "speed_tier": 2,
        "quality_tier": 4,
        "gpu_required": False,
        "status": "ready",
        "variant": "eccv16",
        "backend": "instcolorization2025",
    },
    {
        "id": "chromanet_v3",
        "name": "ChromaNet v3",
        "description": "Custom architecture with temporal attention and scene memory.",
        "architecture": "Encoder-Decoder + Attention",
        "speed_tier": 1,
        "quality_tier": 5,
        "gpu_required": True,
        "status": "needs_checkpoint",
        "variant": None,
        "backend": "colorize_chromanet_v3",
    },
    {
        "id": "enhanced_zhang_bebo",
        "name": "Enhanced Zhang",
        "description": "Bebo's experimental architecture with improved local features.",
        "architecture": "CNN — LAB colorspace",
        "speed_tier": 3,
        "quality_tier": 4,
        "gpu_required": False,
        "status": "ready",
        "variant": None,
        "backend": "Enhanced Zhang (Bebo's Experiment)",
    },
]

def _check_model_statuses():
    checkpoint = ROOT / "ChromaNet_v3_complete" / "chromanet_v3" / "checkpoints" / "checkpoint_latest.pth"
    for m in MODEL_REGISTRY:
        if m["id"] == "chromanet_v3":
            m["status"] = "ready" if checkpoint.exists() else "needs_checkpoint"


# ─────────────────────────────────────────────────────────────────────────────
#  Project history
# ─────────────────────────────────────────────────────────────────────────────
HISTORY_PATH = GUI_DIR / "history.json"

def _load_history() -> list:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    return []

def _save_history(history: list):
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

def _add_to_history(entry: dict):
    history = _load_history()
    history.insert(0, entry)
    history = history[:50]  # keep last 50
    _save_history(history)


# ─────────────────────────────────────────────────────────────────────────────
#  Preset management
# ─────────────────────────────────────────────────────────────────────────────
PRESETS_DIR = GUI_DIR / "presets"
PRESETS_DIR.mkdir(exist_ok=True)

BUILTIN_PRESETS = [
    {
        "id": "fast_preview",
        "name": "Fast Preview",
        "builtin": True,
        "model": "colorize_zhang_eccv16",
        "input_size": 128,
        "batch_size": 24,
        "smoothing_enabled": False,
        "smoothing_window": 5,
        "smoothing_anchor": 0.65,
        "extraction_format": "jpg",
        "codec": "h264",
        "speed_stars": 4,
        "quality_stars": 2,
        "description": "Quick evaluation and rough drafts. ~25 fps",
    },
    {
        "id": "balanced",
        "name": "Balanced",
        "builtin": True,
        "model": "colorize_zhang_siggraph17",
        "input_size": 256,
        "batch_size": 12,
        "smoothing_enabled": True,
        "smoothing_window": 9,
        "smoothing_anchor": 0.65,
        "extraction_format": "jpg",
        "codec": "h264",
        "speed_stars": 3,
        "quality_stars": 3,
        "description": "General purpose, recommended for most videos. ~12 fps",
    },
    {
        "id": "high_quality",
        "name": "High Quality",
        "builtin": True,
        "model": "colorize_zhang_siggraph17",
        "input_size": 512,
        "batch_size": 6,
        "smoothing_enabled": True,
        "smoothing_window": 13,
        "smoothing_anchor": 0.60,
        "extraction_format": "png",
        "codec": "h265",
        "speed_stars": 2,
        "quality_stars": 4,
        "description": "Archival work and final delivery. ~6 fps",
    },
    {
        "id": "extreme_quality",
        "name": "Extreme Quality",
        "builtin": True,
        "model": "chromanet_v3",
        "input_size": 512,
        "batch_size": 4,
        "smoothing_enabled": True,
        "smoothing_window": 21,
        "smoothing_anchor": 0.55,
        "extraction_format": "png",
        "codec": "h265",
        "speed_stars": 1,
        "quality_stars": 5,
        "description": "Maximum quality, requires ChromaNet v3 checkpoint + CUDA. ~3 fps",
    },
]

def _load_presets() -> list:
    custom = []
    for f in PRESETS_DIR.glob("*.json"):
        try:
            custom.append(json.loads(f.read_text()))
        except Exception:
            pass
    return BUILTIN_PRESETS + custom


# ─────────────────────────────────────────────────────────────────────────────
#  Processing engine — fully non-interactive, drives real backend directly
# ─────────────────────────────────────────────────────────────────────────────
_active_process: subprocess.Popen | None = None
_process_lock = threading.Lock()
_pause_flag = threading.Event()
_pause_flag.set()
_cancel_flag = threading.Event()   # set = cancel requested
_event_queue: queue.Queue = queue.Queue()


def _extract_frames_headless(ffmpeg_path: str, video_path: Path, temp_root: Path,
                              fmt: str = "jpg") -> Path:
    """Non-interactive frame extraction — format comes from GUI, no input() prompt."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = temp_root / timestamp / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if fmt == "jpg" else "png"
    cmd = [
        ffmpeg_path,
        "-hide_banner", "-v", "error", "-stats",
        "-i", str(video_path),
        "-q:v", "2",
        str(output_dir / f"frame_%05d.{ext}"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_dir


def _run_pipeline(job: dict, window: webview.Window):
    """
    Drives the REAL VideoRasterization backend from the GUI.
    All settings come from the GUI job config — zero input() prompts.
    Emits structured JSON events for live UI updates.
    Supports pause (via _pause_flag) and cancel (via _cancel_flag).
    """
    sys.path.insert(0, str(ROOT))
    _cancel_flag.clear()

    def emit(event_type: str, **kwargs):
        payload = {"event": event_type, **kwargs}
        try:
            window.evaluate_js(f"window.__chromaEvent({json.dumps(payload)})")
        except Exception:
            pass

    def check_cancel():
        if _cancel_flag.is_set():
            raise InterruptedError("Cancelled")
        _pause_flag.wait()  # blocks here while paused

    try:
        import imageio_ffmpeg

        # All settings from GUI — no prompts needed
        video_path    = Path(job["video_path"])
        fmt           = job.get("extraction_format", "jpg")
        model_name    = job.get("model_backend", "colorize_zhang")
        model_variant = job.get("model_variant", "siggraph17")
        batch_size    = int(job.get("batch_size", 12))
        input_size    = int(job.get("input_size", 256))
        smoothing_on  = bool(job.get("smoothing_enabled", True))
        smooth_win    = int(job.get("smoothing_window", 9))
        smooth_anch   = float(job.get("smoothing_anchor", 0.65))
        codec         = job.get("codec", "h264")
        fps           = int(float(job.get("fps") or 24))
        use_gpu       = HAS_TORCH and (torch.cuda.is_available() if HAS_TORCH else False)
        n_threads     = os.cpu_count() or 8

        temp_root = ROOT / "temp"
        temp_root.mkdir(exist_ok=True)

        # ── STAGE 1: Frame Extraction (headless) ──────────────────────────
        emit("stage_start", stage="extraction", label="Extracting frames")
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        frames_dir = _extract_frames_headless(ffmpeg_path, video_path, temp_root, fmt=fmt)
        check_cancel()

        frame_files = sorted(frames_dir.glob(f"frame_*.{fmt}"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("frame_*.png")) + sorted(frames_dir.glob("frame_*.jpg"))
        total_frames = len(frame_files)
        if total_frames == 0:
            emit("error", message="No frames extracted — check your video file.")
            return
        emit("stage_complete", stage="extraction", total_frames=total_frames,
             label=f"Extracted {total_frames:,} frames")

        # ── STAGE 2: AI Colorization with live progress ───────────────────
        emit("stage_start", stage="colorization",
             label=f"Colorizing with {job.get('model_id','model')}", total=total_frames)

        color_dir = frames_dir.parent / f"{frames_dir.name}_colorized"

        _col_error = [None]
        _col_done  = threading.Event()

        def _do_colorize():
            try:
                from video_pipeline.colorization import run_colorization
                opts = {
                    "batch_size": batch_size,
                    "input_size": input_size,
                    "variant": model_variant,
                    "style": "eccv16"
                }
                run_colorization(frames_dir, model_name, use_gpu, **opts)
            except Exception as e:
                _col_error[0] = e
            finally:
                _col_done.set()

        col_thread = threading.Thread(target=_do_colorize, daemon=True)
        col_thread.start()

        # Poll output dir to emit per-frame progress
        t0 = time.time()
        last_done = 0
        fps_samples: list[float] = []

        while not _col_done.is_set():
            check_cancel()
            done = len(list(color_dir.glob(f"*.{fmt}"))) or \
                   len(list(color_dir.glob("*.jpg"))) + len(list(color_dir.glob("*.png")))

            if done > last_done:
                elapsed = max(time.time() - t0, 0.001)
                fps_now = done / elapsed
                fps_samples.append(fps_now)
                avg_fps = sum(fps_samples[-10:]) / len(fps_samples[-10:])
                eta = max(total_frames - done, 0) / max(avg_fps, 0.001)

                latest = sorted(color_dir.glob(f"*.{fmt}")) or \
                         sorted(color_dir.glob("*.jpg")) + sorted(color_dir.glob("*.png"))
                
                preview_b64 = None
                if latest:
                    import base64
                    try:
                        # Grab the second-to-last file to avoid reading a file that is currently being written to by PIL
                        target_file = latest[-2] if len(latest) > 1 else latest[-1]
                        ext = target_file.suffix.lower().strip('.')
                        mime = f"image/{ext}" if ext in ('png', 'jpeg', 'jpg') else "image/jpeg"
                        
                        # Small sleep to ensure the file handle is flushed if it's the only file
                        if len(latest) == 1:
                            time.sleep(0.05)
                            
                        data = base64.b64encode(target_file.read_bytes()).decode()
                        preview_b64 = f"data:{mime};base64,{data}"
                    except Exception:
                        pass

                emit("frame_processed",
                     frame_index=done, total_frames=total_frames,
                     fps=round(fps_now, 1), avg_fps=round(avg_fps, 1),
                     elapsed_seconds=int(elapsed), eta_seconds=int(eta),
                     preview_path=preview_b64)
                last_done = done
            time.sleep(0.5)

        if _col_error[0]:
            raise _col_error[0]
        check_cancel()
        emit("stage_complete", stage="colorization")

        # ── STAGE 3: Temporal Smoothing ───────────────────────────────────
        emit("stage_start", stage="smoothing",
             label=f"Temporal smoothing (window={smooth_win})" if smoothing_on else "Smoothing skipped")
        smooth_dir = None
        if smoothing_on and smooth_win >= 3:
            from video_pipeline.smoothing import apply_temporal_smoothing_step
            opts = {"enabled": True, "window": smooth_win, "anchor": smooth_anch}
            smooth_dir = apply_temporal_smoothing_step(frames_dir, color_dir, opts)
            check_cancel()
        emit("stage_complete", stage="smoothing")

        # ── STAGE 4: Colorization Report ──────────────────────────────────
        emit("stage_start", stage="reporting", label="Generating report")
        try:
            from tools.preview_report import generate_report as _gen_report
            reports_dir = ROOT / "reports"
            reports_dir.mkdir(exist_ok=True)
            _gen_report(
                frames_gray_dir=frames_dir,
                frames_color_dir=color_dir,
                out_png=reports_dir / "report.png",
                out_json=reports_dir / "report.json",
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Report generation failed: {e}")
        emit("stage_complete", stage="reporting")


        # ── STAGE 5: Video Encoding ───────────────────────────────────────
        emit("stage_start", stage="encoding", label="Encoding output video")
        from video_pipeline.reconstruction import rebuild_video_output
        output_path = rebuild_video_output(color_dir, smooth_dir, video_path, fps)
        check_cancel()
        # Copy to GUI folder for native webview playback
        preview_mp4 = SRC_DIR / "output_preview.mp4"
        try:
            if preview_mp4.exists(): preview_mp4.unlink()
            import shutil
            shutil.copyfile(output_path, preview_mp4)
            preview_vid = "output_preview.mp4"
        except Exception:
            preview_vid = None

        emit("stage_complete", stage="encoding")
        emit("pipeline_complete", output_path=str(output_path), preview_video=preview_vid)

        # ── History ───────────────────────────────────────────────────────
        _add_to_history({
            "id": str(int(time.time())),
            "source_path": str(video_path),
            "output_path": str(output_path),
            "model": job.get("model_id"),
            "preset": job.get("preset_id"),
            "status": "completed",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_frames": total_frames,
        })
        emit("pipeline_complete", output_path=str(output_path))

    except InterruptedError:
        emit("pipeline_cancelled")
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit("error", message=str(e), detail=traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  API exposed to JS (via pywebview)
# ─────────────────────────────────────────────────────────────────────────────
class ChromaAPI:
    def __init__(self):
        self._window: webview.Window | None = None
        self._processing_thread: threading.Thread | None = None

    def set_window(self, window: webview.Window):
        self._window = window

    # ── System ──────────────────────────────────────────────────────────────
    def get_hardware_summary(self) -> dict:
        _check_model_statuses()
        return _get_hardware_summary()

    def get_system_metrics(self) -> dict:
        return _get_system_metrics()

    # ── Models ───────────────────────────────────────────────────────────────
    def get_models(self) -> list:
        _check_model_statuses()
        return MODEL_REGISTRY

    # ── Presets ──────────────────────────────────────────────────────────────
    def get_presets(self) -> list:
        return _load_presets()

    def save_preset(self, preset: dict) -> dict:
        preset_id = preset.get("id") or f"custom_{int(time.time())}"
        preset["id"] = preset_id
        preset["builtin"] = False
        path = PRESETS_DIR / f"{preset_id}.json"
        path.write_text(json.dumps(preset, indent=2))
        return {"ok": True, "id": preset_id}

    def delete_preset(self, preset_id: str) -> dict:
        path = PRESETS_DIR / f"{preset_id}.json"
        if path.exists():
            path.unlink()
            return {"ok": True}
        return {"ok": False, "error": "not found"}

    # ── History ──────────────────────────────────────────────────────────────
    def get_history(self) -> list:
        return _load_history()

    def clear_history(self) -> dict:
        _save_history([])
        return {"ok": True}

    # ── File Operations ───────────────────────────────────────────────────────
    def open_file_dialog(self) -> str | None:
        try:
            dialog_type = webview.FileDialog.OPEN
        except AttributeError:
            dialog_type = webview.OPEN_DIALOG
            
        result = self._window.create_file_dialog(
            dialog_type,
            allow_multiple=False,
            file_types=("Video Files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v)",),
        )
        return result[0] if result else None

    def open_folder_dialog(self) -> str | None:
        try:
            dialog_type = webview.FileDialog.FOLDER
        except AttributeError:
            dialog_type = webview.FOLDER_DIALOG
            
        result = self._window.create_file_dialog(dialog_type)
        return result[0] if result else None

    def open_output_in_explorer(self, path: str):
        if path and Path(path).exists():
            import os
            os.startfile(Path(path).parent)

    def play_output_video(self, path: str):
        if path and Path(path).exists():
            import os
            os.startfile(path)
        return {"ok": True}

    def open_in_explorer(self, path: str) -> dict:
        import subprocess as sp
        p = Path(path)
        if p.is_file():
            sp.Popen(["explorer", "/select,", str(p)])
        elif p.is_dir():
            sp.Popen(["explorer", str(p)])
        return {"ok": True}

    def get_video_info(self, path: str) -> dict:
        """Return basic video metadata using FFmpeg."""
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            result = subprocess.run(
                [ffmpeg, "-i", path],
                capture_output=True, text=True, timeout=10
            )
            output = result.stderr
            info: dict[str, Any] = {"path": path, "name": Path(path).name}
            # parse duration
            for line in output.splitlines():
                if "Duration:" in line:
                    dur = line.split("Duration:")[1].split(",")[0].strip()
                    info["duration_str"] = dur
                if "Stream" in line and "Video:" in line:
                    # fps
                    import re
                    fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
                    if fps_m:
                        info["fps"] = float(fps_m.group(1))
                    # resolution
                    res_m = re.search(r"(\d{3,5})x(\d{3,5})", line)
                    if res_m:
                        info["width"] = int(res_m.group(1))
                        info["height"] = int(res_m.group(2))
            return info
        except Exception as e:
            return {"path": path, "name": Path(path).name, "error": str(e)}

    def get_first_frame(self, video_path: str) -> str | None:
        """Extract first frame, return base64 data URI."""
        try:
            import imageio_ffmpeg, base64
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            out_path = GUI_DIR / "src" / "_preview_frame.jpg"
            subprocess.run([
                ffmpeg, "-y", "-i", video_path,
                "-vframes", "1", "-q:v", "3",
                str(out_path)
            ], capture_output=True, timeout=15)
            if out_path.exists():
                data = base64.b64encode(out_path.read_bytes()).decode()
                return f"data:image/jpeg;base64,{data}"
        except Exception:
            pass
        return None

    def get_report(self) -> dict | None:
        """Return the latest colorization report JSON."""
        report_json = ROOT / "reports" / "report.json"
        report_png = ROOT / "reports" / "report.png"
        if report_json.exists():
            try:
                data = json.loads(report_json.read_text())
                if report_png.exists():
                    import base64
                    data["report_png_b64"] = "data:image/png;base64," + base64.b64encode(report_png.read_bytes()).decode()
                return data
            except Exception:
                pass
        return None

    # ── Processing ────────────────────────────────────────────────────────────
    def start_processing(self, job: dict) -> dict:
        if self._processing_thread and self._processing_thread.is_alive():
            return {"ok": False, "error": "Already processing"}
        _pause_flag.set()
        self._processing_thread = threading.Thread(
            target=_run_pipeline, args=(job, self._window), daemon=True
        )
        self._processing_thread.start()
        return {"ok": True}

    def pause_processing(self) -> dict:
        _pause_flag.clear()
        return {"ok": True}

    def resume_processing(self) -> dict:
        _pause_flag.set()
        return {"ok": True}

    def cancel_processing(self) -> dict:
        _cancel_flag.set()   # signals the pipeline to stop at next check_cancel()
        _pause_flag.set()    # un-pause so the loop can reach check_cancel()
        return {"ok": True}

    def is_processing(self) -> bool:
        return bool(self._processing_thread and self._processing_thread.is_alive())


# ─────────────────────────────────────────────────────────────────────────────
#  Entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    api = ChromaAPI()
    index_path = SRC_DIR / "index.html"

    window = webview.create_window(
        title="VideoRasterization",
        url=str(index_path),
        js_api=api,
        width=1440,
        height=900,
        min_size=(1024, 680),
        background_color="#0E0E0E",
        text_select=False,
    )

    api.set_window(window)

    def on_loaded():
        # prime hardware summary immediately
        hw = api.get_hardware_summary()
        window.evaluate_js(f"window.__chromaEvent({json.dumps({'event':'hardware_ready','data':hw})})")
        
        try:
            from webview.dom import DOMEventHandler
            body = window.dom.get_element('body')
            if body:
                body.on('dragover', DOMEventHandler(lambda e: None, prevent_default=True))
                body.on('drop', DOMEventHandler(lambda e: None, prevent_default=True))
                
            drop_zone = window.dom.get_element('#drop-zone')
            if drop_zone:
                def on_drop(e):
                    files = e.get('dataTransfer', {}).get('files', [])
                    for f in files:
                        path = f.get('pywebviewFullPath')
                        if path and any(path.lower().endswith(ext) for ext in ['.mp4','.mov','.mkv','.avi','.webm','.m4v']):
                            window.evaluate_js(f'CS.loadVideo(String.raw`{path}`)')
                            break
                
                drop_zone.on('dragover', DOMEventHandler(lambda e: None, prevent_default=True))
                drop_zone.on('drop', DOMEventHandler(on_drop, prevent_default=True))
        except Exception as e:
            print("Failed to setup drag and drop:", e)

    window.events.loaded += on_loaded
    webview.start(debug=False)


if __name__ == "__main__":
    main()
