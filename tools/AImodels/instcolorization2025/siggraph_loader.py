from pathlib import Path
import shutil

try:
    import requests  # type: ignore
except Exception:
    requests = None

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR.parent / "DataSets"
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
LOCAL_DIR = BASE_DIR / "checkpoints" / "base"

SIGGRAPH17_URL = "https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth"
ECCV16_URL = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _find_existing(names):
    for name in names:
        cache_path = CACHE_DIR / name
        local_path = LOCAL_DIR / name
        dataset_path = DATASETS_DIR / name
        if dataset_path.is_file():
            return dataset_path
        if cache_path.is_file():
            return cache_path
        if local_path.is_file():
            return local_path
    return None


def _download(url: str, dest: Path):
    if requests is None:
        raise RuntimeError("requests is required to download model weights.")
    _ensure_dir(dest.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    if not dest.exists() or dest.stat().st_size < 1024 * 100:
        raise IOError(f"Downloaded file looks invalid: {dest}")
    cache_dest = CACHE_DIR / dest.name
    if not cache_dest.exists():
        _ensure_dir(cache_dest.parent)
        try:
            shutil.copy2(dest, cache_dest)
        except OSError:
            pass
    return dest


def _get_weight_path(model_name: str, url: str):
    official_name = Path(url).name
    candidates = [official_name, f"{model_name}.pth"]
    existing = _find_existing(candidates)
    if existing:
        return existing
    dest = LOCAL_DIR / f"{model_name}.pth"
    return _download(url, dest)


def load_siggraph17() -> Path:
    """Return Path to SIGGRAPH17 weights (cache -> local -> auto-download)."""
    return _get_weight_path("siggraph17", SIGGRAPH17_URL)


def load_eccv16() -> Path:
    """Return Path to ECCV16 weights (cache -> local -> auto-download)."""
    return _get_weight_path("eccv16", ECCV16_URL)
