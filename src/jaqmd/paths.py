import os
from pathlib import Path


def db_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    d = base / "jaqmd"
    d.mkdir(parents=True, exist_ok=True)
    return d / "index.sqlite"
