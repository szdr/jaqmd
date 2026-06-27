import pytest


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
