"""env / 設定ファイルの読み込みと優先順位解決。

優先順位: CLI 引数 > 環境変数 > 設定ファイル (TOML) > ハードコード既定。
CLI 引数との合成は呼び出し側（cli.py 等）が行う。ここでは
「環境変数 > 設定ファイル > 既定」までを解決した `settings` を提供する。

設定ファイルの場所は `$XDG_CONFIG_HOME/jaqmd/config.toml`
（`XDG_CONFIG_HOME` 未設定時は `~/.config/jaqmd/config.toml`）。
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "jaqmd" / "config.toml"


def cache_base() -> Path:
    """jaqmd のキャッシュルート（DB・モデルの既定置き場）。XDG_CACHE_HOME を尊重する。

    `settings` には含めない（プロセス起動後の env 変更にも追従させるため、
    呼び出しごとに毎回 os.environ を読む）。
    """
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "jaqmd"


def _load_toml() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(
            f"警告: 設定ファイルの読み込みに失敗しました（{path}）: {e}",
            file=sys.stderr,
        )
        return {}


def _raw(table: dict, section: str, key: str, env: str) -> Any:
    """env > config[section][key] の優先で生値を返す（どちらも未設定なら None）。"""
    if env in os.environ:
        return os.environ[env]
    return table.get(section, {}).get(key)


def _as_bool(value: Any, env: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    print(
        f"警告: {env} の値が不正です（{value!r}）。既定値 {default} を使用します。",
        file=sys.stderr,
    )
    return default


def _as_int(value: Any, env: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        print(
            f"警告: {env} の値が不正です（{value!r}）。既定値 {default} を使用します。",
            file=sys.stderr,
        )
        return default


def _as_float(value: Any, env: str, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        print(
            f"警告: {env} の値が不正です（{value!r}）。既定値 {default} を使用します。",
            file=sys.stderr,
        )
        return default


def _as_str(value: Any, default: str) -> str:
    return str(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    # [search]
    search_n: int = 5
    search_format: str = "plain"  # plain|json|md|xml|files
    search_full: bool = False
    search_min_score: Optional[float] = None
    search_reranker: str = "default"
    search_rerank: bool = True
    search_qe: bool = True
    search_snippet_chars: int = 300
    # [index]
    index_glob: str = "**/*.md"
    index_batch_size: int = 1
    # [general]
    quiet: bool = False
    # [paths]（None なら cache_base() 基準の既定値にフォールバック）
    db_path: Optional[str] = None
    models_dir: Optional[str] = None
    # [models]
    embed_model: str = "sirasagi62/ruri-v3-310m-ONNX"
    reranker_model: str = "szdr/ruri-v3-reranker-310m-onnx"
    qe_repo: str = "szdr/jaqmd-qe-gemma-4-e2b-it"
    # [tuning]
    rrf_k: int = 60
    rerank_top_k: int = 50
    rerank_candidate_limit: int = 40


def _build() -> Settings:
    table = _load_toml()
    d = Settings()

    def g(section: str, key: str, env: str) -> Any:
        return _raw(table, section, key, env)

    return Settings(
        search_n=_as_int(
            g("search", "n", "JAQMD_SEARCH_N"), "JAQMD_SEARCH_N", d.search_n
        ),
        search_format=_as_str(
            g("search", "format", "JAQMD_SEARCH_FORMAT"), d.search_format
        ),
        search_full=_as_bool(
            g("search", "full", "JAQMD_SEARCH_FULL"), "JAQMD_SEARCH_FULL", d.search_full
        ),
        search_min_score=_as_float(
            g("search", "min_score", "JAQMD_SEARCH_MIN_SCORE"),
            "JAQMD_SEARCH_MIN_SCORE",
            d.search_min_score,
        ),
        search_reranker=_as_str(
            g("search", "reranker", "JAQMD_SEARCH_RERANKER"), d.search_reranker
        ),
        search_rerank=_as_bool(
            g("search", "rerank", "JAQMD_SEARCH_RERANK"),
            "JAQMD_SEARCH_RERANK",
            d.search_rerank,
        ),
        search_qe=_as_bool(
            g("search", "qe", "JAQMD_SEARCH_QE"), "JAQMD_SEARCH_QE", d.search_qe
        ),
        search_snippet_chars=_as_int(
            g("search", "snippet_chars", "JAQMD_SEARCH_SNIPPET_CHARS"),
            "JAQMD_SEARCH_SNIPPET_CHARS",
            d.search_snippet_chars,
        ),
        index_glob=_as_str(g("index", "glob", "JAQMD_INDEX_GLOB"), d.index_glob),
        index_batch_size=_as_int(
            g("index", "batch_size", "JAQMD_INDEX_BATCH_SIZE"),
            "JAQMD_INDEX_BATCH_SIZE",
            d.index_batch_size,
        ),
        quiet=_as_bool(g("general", "quiet", "JAQMD_QUIET"), "JAQMD_QUIET", d.quiet),
        db_path=_raw(table, "paths", "db", "JAQMD_DB_PATH"),
        models_dir=_raw(table, "paths", "models", "JAQMD_MODELS_DIR"),
        embed_model=_as_str(g("models", "embed", "JAQMD_MODELS_EMBED"), d.embed_model),
        reranker_model=_as_str(
            g("models", "reranker", "JAQMD_MODELS_RERANKER"), d.reranker_model
        ),
        qe_repo=_as_str(g("models", "qe_repo", "JAQMD_MODELS_QE_REPO"), d.qe_repo),
        rrf_k=_as_int(
            g("tuning", "rrf_k", "JAQMD_TUNING_RRF_K"), "JAQMD_TUNING_RRF_K", d.rrf_k
        ),
        rerank_top_k=_as_int(
            g("tuning", "rerank_top_k", "JAQMD_TUNING_RERANK_TOP_K"),
            "JAQMD_TUNING_RERANK_TOP_K",
            d.rerank_top_k,
        ),
        rerank_candidate_limit=_as_int(
            g(
                "tuning",
                "rerank_candidate_limit",
                "JAQMD_TUNING_RERANK_CANDIDATE_LIMIT",
            ),
            "JAQMD_TUNING_RERANK_CANDIDATE_LIMIT",
            d.rerank_candidate_limit,
        ),
    )


settings = _build()


def reload() -> Settings:
    """env / 設定ファイルを再読み込みし、`settings` を再構築する。

    `settings` はモジュール初回 import 時に一度だけ構築されるため、
    プロセス起動後に env や設定ファイルを変更した場合（主にテスト）は
    このモジュール自身の `settings` 属性は更新されるが、既に
    `from .config import settings` の形で値を束縛済みのモジュール側の
    参照は更新されない点に注意（Tier2 の定数は各モジュール import 時点で
    確定する設計のため）。
    """
    global settings
    settings = _build()
    return settings
