import argparse
from pathlib import Path

from videocolor.frames import extract_frames, extract_audio
# stubs for next steps:
def colorize_dir(frames_dir: Path, out_dir: Path, models_dir: Path, model: str = "opencv_zhang") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"(todo) colorize_dir -> {out_dir} using {model}")

def smooth_sequence(color_dir: Path, alpha: float = 0.7) -> None:
    print(f"(todo) smooth_sequence alpha={alpha}")

def build_video(color_dir: Path, audio_path: Path|None, out_path: Path, fps: int=24, encoder: str|None=None) -> None:
    print(f"(todo) build_video -> {out_path} @ {fps}fps encoder={encoder}")

def main():
    p = argparse.ArgumentParser(prog="videocolor", description="AI Video Colorization – terminal pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract frames and audio")
    pe.add_argument("input", type=Path)
    pe.add_argument("--frames-dir", type=Path, default=Path("frames"))
    pe.add_argument("--audio", type=Path, default=Path("audio.aac"))

    pc = sub.add_parser("colorize", help="Colorize frames")
    pc.add_argument("--frames-dir", type=Path, default=Path("frames"))
    pc.add_argument("--out-dir", type=Path, default=Path("color"))
    pc.add_argument("--model", choices=["opencv_zhang","deoldify"], default="opencv_zhang")
    pc.add_argument("--models-dir", type=Path, default=Path("models"))

    ps = sub.add_parser("smooth", help="Temporal smoothing")
    ps.add_argument("--color-dir", type=Path, default=Path("color"))
    ps.add_argument("--alpha", type=float, default=0.7)

    pr = sub.add_parser("rebuild", help="Rebuild video")
    pr.add_argument("--color-dir", type=Path, default=Path("color"))
    pr.add_argument("--audio", type=Path, default=Path("audio.aac"))
    pr.add_argument("--out", type=Path, default=Path("output.mp4"))
    pr.add_argument("--fps", type=int, default=24)
    pr.add_argument("--encoder", choices=["h264_nvenc","h264_amf","h264_qsv","libx264"], default=None)

    pall = sub.add_parser("all", help="extract → colorize → smooth → rebuild")
    pall.add_argument("input", type=Path)
    pall.add_argument("--frames-dir", type=Path, default=Path("frames"))
    pall.add_argument("--audio", type=Path, default=Path("audio.aac"))
    pall.add_argument("--color-dir", type=Path, default=Path("color"))
    pall.add_argument("--out", type=Path, default=Path("output.mp4"))
    pall.add_argument("--fps", type=int, default=24)
    pall.add_argument("--model", choices=["opencv_zhang","deoldify"], default="opencv_zhang")
    pall.add_argument("--models-dir", type=Path, default=Path("models"))
    pall.add_argument("--alpha", type=float, default=0.7)
    pall.add_argument("--skip-smooth", action="store_true")
    pall.add_argument("--encoder", choices=["h264_nvenc","h264_amf","h264_qsv","libx264"], default=None)

    args = p.parse_args()

    if args.cmd=="extract":
        print(f"[extract] video={args.input}")
        extract_frames(args.input, args.frames_dir)
        extract_audio(args.input, args.audio)

    elif args.cmd=="colorize":
        colorize_dir(args.frames_dir, args.out_dir, args.models_dir, model=args.model)

    elif args.cmd=="smooth":
        smooth_sequence(args.color_dir, alpha=args.alpha)

    elif args.cmd=="rebuild":
        build_video(args.color_dir, args.audio, args.out, fps=args.fps, encoder=args.encoder)

    elif args.cmd=="all":
        extract_frames(args.input, args.frames_dir)
        extract_audio(args.input, args.audio)
        colorize_dir(args.frames_dir, args.color_dir, args.models_dir, model=args.model)
        if not args.skip_smooth:
            smooth_sequence(args.color_dir, alpha=args.alpha)
        build_video(args.color_dir, args.audio, args.out, fps=args.fps, encoder=args.encoder)

if __name__ == "__main__":
    main()
