import subprocess
from datetime import datetime

from tools.console import status, error

def extract_frames(ffmpeg_path, video_path, temp_root):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #for the folder creation
    output_dir = temp_root / timestamp / "frames"   #creates a folder in this format
    output_dir.mkdir(parents=True, exist_ok=True)

    # ask for quality mode
    print("Choose extraction mode:")
    print("1 = Full Quality (PNG, large size)")
    print("2 = Faster Encoding (JPG, smaller size)")
    choice = input("> ").strip()

    if choice == "2":
        ext = "jpg"
        codec = "mjpeg"
        status("[mode] faster JPG extraction selected.")
    else:
        ext = "png"
        codec = "png"
        status("[mode] full-quality PNG extraction selected.")

    cmd = [
        ffmpeg_path,
        "-hide_banner", "-v", "error",
        "-i", str(video_path),
        "-q:v", "2",  # good quality for JPG, ignored for PNG
        str(output_dir / f"frame_%05d.{ext}")
    ]

    status(f"[extract] saving frames to: {output_dir}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stderr:
            error(result.stderr.strip())
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    status("[ok] frame extraction complete.")
    return output_dir

