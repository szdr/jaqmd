import pytest


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
