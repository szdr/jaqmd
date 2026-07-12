from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from jaqmd.store import add_collection, set_meta, upsert_document


@pytest.fixture
def trigram_conn(conn, doc_dir):
    """trigram インデックスのみ構築済みの接続（test_query.py の同名フィクスチャと同構成）。"""
    add_collection(conn, "test", str(doc_dir))
    upsert_document(
        conn,
        collection="test",
        path="a.md",
        body="形態素解析は日本語の自然言語処理の基礎技術です",
        title="形態素解析について",
        mtime=1000,
    )
    upsert_document(
        conn,
        collection="test",
        path="b.md",
        body="検索エンジンの仕組みと実装方法を解説します",
        title="検索エンジン入門",
        mtime=1001,
    )
    conn.commit()
    set_meta(conn, "trigram_indexed", "1")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------


def test_run_query_requires_trigram_index(conn):
    from jaqmd.mcp.server import run_query

    with pytest.raises(ValueError, match="update"):
        run_query(conn, [("lex", "テスト")])


def test_run_query_returns_json_shape(trigram_conn):
    from jaqmd.mcp.server import run_query

    results = run_query(trigram_conn, [("lex", "形態素解析")])
    assert len(results) >= 1
    r = results[0]
    assert set(r.keys()) == {"docid", "score", "filepath", "title", "snippet"}
    assert any("a.md" in r["filepath"] for r in results)


def test_run_query_empty_searches_raises(trigram_conn):
    from jaqmd.mcp.server import run_query

    with pytest.raises(ValueError):
        run_query(trigram_conn, [])


def test_run_query_too_many_searches_raises(trigram_conn):
    from jaqmd.mcp.server import run_query

    with pytest.raises(ValueError):
        run_query(trigram_conn, [("lex", "x")] * 11)


def test_run_query_collections_filter(conn, tmp_path):
    from jaqmd.mcp.server import run_query

    d1, d2 = tmp_path / "col1", tmp_path / "col2"
    d1.mkdir()
    d2.mkdir()
    add_collection(conn, "col1", str(d1))
    add_collection(conn, "col2", str(d2))
    upsert_document(
        conn,
        collection="col1",
        path="a.md",
        body="日本語処理の解説",
        title="A",
        mtime=1000,
    )
    upsert_document(
        conn,
        collection="col2",
        path="b.md",
        body="日本語処理は重要です",
        title="B",
        mtime=1001,
    )
    conn.commit()
    set_meta(conn, "trigram_indexed", "1")
    conn.commit()

    results = run_query(conn, [("lex", "日本語")], collections=["col1"])
    assert results
    assert all(r["filepath"].startswith("col1/") for r in results)


def test_run_query_limit(trigram_conn):
    from jaqmd.mcp.server import run_query

    results = run_query(trigram_conn, [("lex", "す")], limit=1)
    assert len(results) <= 1


# ---------------------------------------------------------------------------
# run_get
# ---------------------------------------------------------------------------


def test_run_get_by_docid(trigram_conn):
    from jaqmd.mcp.server import run_get

    row = trigram_conn.execute(
        "SELECT docid FROM documents WHERE path = 'a.md'"
    ).fetchone()
    result = run_get(trigram_conn, row["docid"])
    assert result["path"] == "a.md"
    assert "形態素解析" in result["body"]


def test_run_get_by_path(trigram_conn):
    from jaqmd.mcp.server import run_get

    result = run_get(trigram_conn, "a.md")
    assert result["docid"]
    assert result["collection"] == "test"


def test_run_get_line_suffix_stripped(trigram_conn):
    from jaqmd.mcp.server import run_get

    result = run_get(trigram_conn, "a.md:10")
    assert result["path"] == "a.md"


def test_run_get_not_found_raises(trigram_conn):
    from jaqmd.mcp.server import run_get

    with pytest.raises(ValueError):
        run_get(trigram_conn, "no/such/doc.md")


# ---------------------------------------------------------------------------
# run_multi_get
# ---------------------------------------------------------------------------


def test_run_multi_get_glob(trigram_conn):
    from jaqmd.mcp.server import run_multi_get

    results = run_multi_get(trigram_conn, "*.md")
    paths = {r["path"] for r in results}
    assert paths == {"a.md", "b.md"}


def test_run_multi_get_comma_separated(trigram_conn):
    from jaqmd.mcp.server import run_multi_get

    results = run_multi_get(trigram_conn, "a.md,b.md")
    paths = {r["path"] for r in results}
    assert paths == {"a.md", "b.md"}


def test_run_multi_get_skips_missing(trigram_conn):
    from jaqmd.mcp.server import run_multi_get

    results = run_multi_get(trigram_conn, "a.md,no/such.md")
    assert len(results) == 1
    assert results[0]["path"] == "a.md"


# ---------------------------------------------------------------------------
# run_status
# ---------------------------------------------------------------------------


def test_run_status_shape(trigram_conn):
    from jaqmd.mcp.server import run_status

    status = run_status(trigram_conn)
    assert status["total_documents"] == 2
    assert status["trigram_count"] >= 2
    assert status["morph_indexed"] is False
    assert status["vec_indexed"] is False
    assert "search" in status["available"]
    assert any(
        c["name"] == "test" and c["documents"] == 2 for c in status["collections"]
    )


def test_run_status_empty_db(conn):
    from jaqmd.mcp.server import run_status

    status = run_status(conn)
    assert status["total_documents"] == 0
    assert status["collections"] == []


# ---------------------------------------------------------------------------
# FastMCP 登録の健全性
# ---------------------------------------------------------------------------


def test_build_server_registers_four_tools():
    import asyncio

    from jaqmd.mcp.server import build_server

    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"query", "get", "multi_get", "status"}
