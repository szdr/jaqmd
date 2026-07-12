import os
import tempfile

import pytest

# jaqmd.config は import 時に一度だけ設定（env / 設定ファイル）を解決するため、
# テスト収集（test_cli.py の `from jaqmd.cli import app` 等）で jaqmd パッケージが
# import される前に、実ユーザーの ~/.config/jaqmd/config.toml や実 env を
# 隔離しておく必要がある。conftest.py のモジュールトップレベルは各テストファイルの
# import より必ず先に実行されるため、ここで隔離する。
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="jaqmd-test-config-")
for _var in (
    "JAQMD_SEARCH_N",
    "JAQMD_SEARCH_FORMAT",
    "JAQMD_SEARCH_FULL",
    "JAQMD_SEARCH_MIN_SCORE",
    "JAQMD_SEARCH_RERANKER",
    "JAQMD_SEARCH_RERANK",
    "JAQMD_SEARCH_QE",
    "JAQMD_INDEX_GLOB",
    "JAQMD_INDEX_BATCH_SIZE",
    "JAQMD_QUIET",
    "JAQMD_DB_PATH",
    "JAQMD_MODELS_DIR",
    "JAQMD_MODELS_EMBED",
    "JAQMD_MODELS_RERANKER",
    "JAQMD_MODELS_QE_REPO",
    "JAQMD_TUNING_RRF_K",
    "JAQMD_TUNING_RERANK_TOP_K",
):
    os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def _no_real_rerank(request, monkeypatch):
    """reranker モデルのロード（1.3GB DL）を回避し、恒等フォールバックさせる。

    integration マーク付きテストのみ実モデルを使わせる。
    """
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr(
        "jaqmd.rerank._get_encoder", lambda model=None, reporter=None: None
    )


@pytest.fixture(autouse=True)
def _no_real_qe(request, monkeypatch):
    """QE モデルのロード（3.4GB DL）を回避し、raw クエリへ degrade させる。

    integration マーク付きテストのみ実モデルを使わせる。
    """
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr("jaqmd.qe._get_llm", lambda reporter=None: None)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    return cache


@pytest.fixture
def conn(tmp_cache):
    from jaqmd.store import connect

    return connect()


@pytest.fixture
def doc_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    return d
