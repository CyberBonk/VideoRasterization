from pathlib import Path

def _clean_path_text(text: str) -> str:
    t = text.strip().strip("'\"")
    if t.lower().startswith("file://"):
        t = t.split("://", 1)[1]
    return t

def get_input_video_path(allowed_exts: set[str]) -> Path:
    while True:
        print("Enter path to input video (drag-and-drop is fine):")
        ptxt = _clean_path_text(input("> "))
        if not ptxt:
            print("[warn] empty input. try again.")
            continue

        p = Path(ptxt).expanduser().resolve()
        if not p.exists():
            print(f"[warn] not found: {p}")
            continue
        if p.is_dir():
            print("[warn] that is a folder. please provide a file.")
            continue
        if p.suffix.lower() not in allowed_exts:
            exts = ", ".join(sorted(allowed_exts))
            print(f"[warn] unsupported extension '{p.suffix}'. allowed: {exts}")
            continue
        return p
