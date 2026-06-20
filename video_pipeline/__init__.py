"""Video pipeline helper package."""

from .colorization import run_colorization
from .env import HAS_IPEX, LOGICAL, ipex
from .frame_extraction import extract_frames
from .input_handling import select_input_video
from .reporting import generate_report
from .reconstruction import rebuild_video_output
from .smoothing import apply_temporal_smoothing_step
from .window_prompt import ask_temporal_smoothing_options

__all__ = [
    "HAS_IPEX",
    "LOGICAL",
    "ipex",
    "ask_temporal_smoothing_options",
    "select_input_video",
    "extract_frames",
    "run_colorization",
    "apply_temporal_smoothing_step",
    "generate_report",
    "rebuild_video_output",
]
