from pathlib import Path

from .config import cache_base, settings


def db_path() -> Path:
    """DB ファイルのパス。`JAQMD_DB_PATH` / 設定ファイル `[paths] db` で上書き可能。"""
    if settings.db_path:
        p = Path(settings.db_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    d = cache_base()
    d.mkdir(parents=True, exist_ok=True)
    return d / "index.sqlite"


def models_dir() -> Path:
    """モデルキャッシュディレクトリ。`JAQMD_MODELS_DIR` / 設定ファイル `[paths] models` で上書き可能。"""
    if settings.models_dir:
        d = Path(settings.models_dir).expanduser().resolve()
    else:
        d = cache_base() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
