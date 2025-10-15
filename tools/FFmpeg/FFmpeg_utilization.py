import subprocess
from datetime import datetime

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
        print("[mode] faster JPG extraction selected.")
    else:
        ext = "png"
        codec = "png"
        print("[mode] full-quality PNG extraction selected.")

    cmd = [
        ffmpeg_path,
        "-hide_banner", "-v", "error", "-stats",
        "-i", str(video_path),
        "-q:v", "2",  # good quality for JPG, ignored for PNG
        str(output_dir / f"frame_%05d.{ext}")
    ]

    print(f"[extract] saving frames to: {output_dir}")
    subprocess.run(cmd, check=True)
    print("[ok] frame extraction complete.")
    return output_dir

