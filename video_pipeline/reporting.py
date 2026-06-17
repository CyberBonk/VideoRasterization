"""Reporting utilities."""

from importlib import import_module
from pathlib import Path
from tools.console import status

ROOT = Path(__file__).resolve().parent.parent
report = import_module("tools.preview_report")


def generate_report(frames_gray_dir: Path, frames_color_dir: Path):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_png = reports_dir / "report.png"
    report_json = reports_dir / "report.json"

    try:
        report.generate_report(
            frames_gray_dir=frames_gray_dir,
            frames_color_dir=frames_color_dir,
            out_png=report_png,
            out_json=report_json,
        )
        status(f"[ok] report saved:\n - {report_png}\n - {report_json}")
    except Exception as e:
        status(f"[warn] report failed: {e}")


__all__ = ["generate_report"]
