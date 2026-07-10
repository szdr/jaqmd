from __future__ import annotations

from pathlib import Path

MODEL_CACHE_DIR = Path.home() / ".cache" / "jaqmd" / "models"


def is_model_cached(repo_id: str, filename: str, cache_dir: str) -> bool:
    """指定モデルファイルが HF hub キャッシュに存在するか判定する。

    huggingface_hub のキャッシュレイアウトを直接参照するため、モデル本体を
    ロードせずに判定できる。未導入・未キャッシュの場合は False を返す。
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir), str)
