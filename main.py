from importlib import import_module
from pathlib import Path
import sys

# ensure the repo root is importable on any machine (no absolute paths)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    # lazy import so tools/ stays swappable; no absolute paths
    input_selector = import_module("tools.input_selector")
    video_path = input_selector.get_input_video_path()
    print(f"[ok] selected video: {video_path}")

if __name__ == "__main__":
    main()
