from __future__ import annotations

import dataclasses

import pytest

from jaqmd.search.query import (
    RRF_K,
    _blend_scores,
    _finalize,
    _rrf_fuse,
    query,
    query_searches,
)
from jaqmd.search.trisearch import SearchResult
from jaqmd.store import add_collection, set_meta, upsert_document

# ---------------------------------------------------------------------------
# _rrf_fuse ユニットテスト（外部依存なし）
# ---------------------------------------------------------------------------


def _make_result(docid: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        docid=docid,
        score=score,
        filepath=f"{docid}.md",
        title=docid,
        snippet="snippet",
        body="body",
    )


def test_rrf_single_list():
    """単一リストの場合、RRF スコアが付与されスコア降順になる。"""
    results = [_make_result("a"), _make_result("b"), _make_result("c")]
    fused = _rrf_fuse([results], top_rank_bonus=False)
    assert [r.docid for r in fused] == ["a", "b", "c"]
    # rank=0: 1/(60+1), rank=1: 1/(60+2), rank=2: 1/(60+3)
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[1].score == pytest.approx(1.0 / (RRF_K + 2))
    assert fused[2].score == pytest.approx(1.0 / (RRF_K + 3))


def test_rrf_fuse_two_lists_overlap():
    """2リストで重複ありの場合、重複 docid のスコアが加算される。"""
    list_a = [_make_result("x"), _make_result("y")]
    list_b = [_make_result("x"), _make_result("z")]
    fused = _rrf_fuse([list_a, list_b], top_rank_bonus=False)
    docids = [r.docid for r in fused]
    # "x" は両リストの rank=0 なので最高スコア
    assert docids[0] == "x"
    assert fused[0].score == pytest.approx(2.0 / (RRF_K + 1))


def test_rrf_fuse_preserves_representative():
    """docid 初出リスト（先頭リスト）の SearchResult が代表として使われる。"""
    rep = _make_result("doc", score=5.0)
    rep2 = dataclasses.replace(rep, score=99.0, snippet="other")
    list_a = [rep]
    list_b = [rep2]
    fused = _rrf_fuse([list_a, list_b], top_rank_bonus=False)
    assert len(fused) == 1
    # score は RRF 計算値（原スコアは使わない）
    expected_score = 2.0 / (RRF_K + 1)
    assert fused[0].score == pytest.approx(expected_score)
    # snippet は先頭リスト（list_a）の代表
    assert fused[0].snippet == "snippet"


def test_rrf_fuse_no_overlap():
    """重複なし: 各 docid の RRF スコアが独立して計算される。"""
    list_a = [_make_result("a")]
    list_b = [_make_result("b")]
    fused = _rrf_fuse([list_a, list_b], top_rank_bonus=False)
    assert len(fused) == 2
    # a, b ともに同じ RRF スコア → どちらが先でもよい
    scores = {r.docid: r.score for r in fused}
    assert scores["a"] == pytest.approx(1.0 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_fuse_empty_lists():
    """空リストを渡しても空を返す。"""
    assert _rrf_fuse([]) == []
    assert _rrf_fuse([[]]) == []
    assert _rrf_fuse([[], []]) == []


def test_rrf_fuse_weights_default_matches_unweighted():
    """weights 未指定時は全リスト重み1.0（従来どおりの挙動）と一致する。"""
    list_a = [_make_result("a"), _make_result("b")]
    list_b = [_make_result("a"), _make_result("c")]
    fused_default = _rrf_fuse([list_a, list_b])
    fused_explicit = _rrf_fuse([list_a, list_b], weights=[1.0, 1.0])
    assert [(r.docid, r.score) for r in fused_default] == [
        (r.docid, r.score) for r in fused_explicit
    ]


def test_rrf_fuse_weights_boost_first_list():
    """先頭リストに weight=2.0 を与えると、そのリスト由来のスコア寄与が2倍になる。"""
    list_a = [_make_result("a")]  # weight 2.0
    list_b = [_make_result("b")]  # weight 1.0
    fused = _rrf_fuse([list_a, list_b], weights=[2.0, 1.0], top_rank_bonus=False)
    scores = {r.docid: r.score for r in fused}
    assert scores["a"] == pytest.approx(2.0 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(1.0 / (RRF_K + 1))
    assert scores["a"] > scores["b"]


def test_rrf_fuse_score_ordering():
    """RRF 後の結果がスコア降順になっている。"""
    # list_a では b→a、list_b では a→b → a と b は異なるスコアになる
    list_a = [_make_result("b"), _make_result("a")]
    list_b = [_make_result("a"), _make_result("b")]
    fused = _rrf_fuse([list_a, list_b])
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# _rrf_fuse: top-rank ボーナス（tobi/qmd 準拠）
# ---------------------------------------------------------------------------


def test_rrf_top_rank_bonus_rank0():
    """いずれかのリストで rank0 に現れた doc に +0.05 が加算される。"""
    results = [_make_result("a"), _make_result("b")]
    fused = _rrf_fuse([results], top_rank_bonus=True)
    scores = {r.docid: r.score for r in fused}
    # a: rank0 → base 1/(60+1) + 0.05
    assert scores["a"] == pytest.approx(1.0 / (RRF_K + 1) + 0.05)
    # b: rank1 → base 1/(60+2) + 0.02（rank<=2）
    assert scores["b"] == pytest.approx(1.0 / (RRF_K + 2) + 0.02)


def test_rrf_top_rank_bonus_rank_gt_2_no_bonus():
    """rank>=3 の doc にはボーナスが付かない。"""
    results = [_make_result(d) for d in ["a", "b", "c", "d"]]
    fused = _rrf_fuse([results], top_rank_bonus=True)
    scores = {r.docid: r.score for r in fused}
    # d: rank3 → ボーナスなし
    assert scores["d"] == pytest.approx(1.0 / (RRF_K + 4))


def test_rrf_top_rank_bonus_uses_min_rank_across_lists():
    """topRank は全リスト中の最小 rank。別リストで rank0 に現れれば +0.05。"""
    list_a = [_make_result("x"), _make_result("y")]  # y は rank1
    list_b = [_make_result("y"), _make_result("z")]  # y は rank0
    fused = _rrf_fuse([list_a, list_b], top_rank_bonus=True)
    scores = {r.docid: r.score for r in fused}
    base_y = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
    assert scores["y"] == pytest.approx(base_y + 0.05)


def test_rrf_top_rank_bonus_default_on():
    """top_rank_bonus のデフォルトは True。"""
    results = [_make_result("a")]
    fused_default = _rrf_fuse([results])
    fused_on = _rrf_fuse([results], top_rank_bonus=True)
    assert fused_default[0].score == pytest.approx(fused_on[0].score)
    assert fused_default[0].score == pytest.approx(1.0 / (RRF_K + 1) + 0.05)


# ---------------------------------------------------------------------------
# _blend_scores: 位置依存ブレンド（tobi/qmd 準拠）
# ---------------------------------------------------------------------------


def test_blend_scores_position_weights():
    """rrfRank のバケット（<=3:0.75 / <=10:0.60 / それ以外:0.40）で重みが切り替わる。"""
    # rerankScore は全件 1.0（差が rrfWeight*rrf_score 部分のみに出る）
    candidates = [_make_result(f"d{i}") for i in range(12)]
    rr = [1.0] * 12
    blended = _blend_scores(candidates, rr)
    by_doc = {r.docid: r.score for r in blended}

    def expected(rank1: int) -> float:
        rrf = 1.0 / rank1
        w = 0.75 if rank1 <= 3 else 0.60 if rank1 <= 10 else 0.40
        return w * rrf + (1.0 - w) * 1.0

    assert by_doc["d0"] == pytest.approx(expected(1))  # rrfRank1, w=0.75
    assert by_doc["d3"] == pytest.approx(expected(4))  # rrfRank4, w=0.60
    assert by_doc["d10"] == pytest.approx(expected(11))  # rrfRank11, w=0.40


def test_blend_scores_degrade_uses_rrf_only():
    """rerank_scores_list=None（degrade）では blended = 1/rrfRank になる。"""
    candidates = [_make_result("a"), _make_result("b"), _make_result("c")]
    blended = _blend_scores(candidates, None)
    # 順序は 1/rrfRank 降順（＝入力順のまま）
    assert [r.docid for r in blended] == ["a", "b", "c"]
    assert blended[0].score == pytest.approx(1.0 / 1)
    assert blended[1].score == pytest.approx(1.0 / 2)
    assert blended[2].score == pytest.approx(1.0 / 3)


def test_blend_scores_sorted_descending():
    """blended スコア降順にソートされる。"""
    candidates = [_make_result("a"), _make_result("b"), _make_result("c")]
    # 下位 doc に高い rerankScore を与えても、位置依存重みで上位保護が効く
    blended = _blend_scores(candidates, [0.0, 0.0, 1.0])
    scores = [r.score for r in blended]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# _finalize: 候補プール切り出し + min_score 足切り
# ---------------------------------------------------------------------------


def test_finalize_candidate_limit_slices_pool():
    """candidate_limit で融合プールが切り出され、それを超える候補は落ちる。"""
    fused = [_make_result(f"d{i}", score=1.0 / (i + 1)) for i in range(10)]
    out = _finalize(
        fused,
        query_for_rerank="q",
        rerank_enabled=False,  # rerank_scores=None → blended=1/rrfRank
        rerank_model="default",
        all_results=False,
        n=100,
        min_score=None,
        candidate_limit=3,
    )
    assert [r.docid for r in out] == ["d0", "d1", "d2"]
    assert out[0].score == pytest.approx(1.0)


def test_finalize_min_score_filters_blended():
    """min_score はブレンド後スコアに対する足切りとして機能する。"""
    fused = [_make_result(f"d{i}") for i in range(5)]
    out = _finalize(
        fused,
        query_for_rerank="q",
        rerank_enabled=False,  # blended=1/rrfRank → d0=1.0, d1=0.5, d2≈0.33...
        rerank_model="default",
        all_results=True,
        n=100,
        min_score=0.5,
        candidate_limit=None,
    )
    # 1/rrfRank >= 0.5 なのは rank1(1.0), rank2(0.5) のみ
    assert [r.docid for r in out] == ["d0", "d1"]


def test_finalize_all_results_caps_rerank_to_candidate_limit(monkeypatch):
    """all_results=True でも reranker には先頭 candidate_limit 件だけ渡す。

    OOM 防止のため全ヒットの一括リランクは行わず、tail は
    rerank_score = 1/rrfRank の degrade 扱いで RRF 順を保つ。
    """
    fused = [_make_result(f"d{i}", score=1.0 / (i + 1)) for i in range(5)]
    reranked_inputs = []

    def fake_rerank_scores(query_text, results, **kwargs):
        reranked_inputs.extend(r.docid for r in results)
        # head 全件に同一スコアを返す（順位への影響を排除して件数だけ検証）
        return [0.5] * len(results)

    monkeypatch.setattr("jaqmd.search.query.rerank_scores", fake_rerank_scores)

    out = _finalize(
        fused,
        query_for_rerank="q",
        rerank_enabled=True,
        rerank_model="default",
        all_results=True,
        n=100,
        min_score=None,
        candidate_limit=2,
    )
    # reranker には head の2件のみ渡された
    assert reranked_inputs == ["d0", "d1"]
    # 出力は全件（all_results）で、tail は degrade（blended = 1/rrfRank）
    assert [r.docid for r in out] == ["d0", "d1", "d2", "d3", "d4"]
    assert out[2].score == pytest.approx(1.0 / 3)
    assert out[3].score == pytest.approx(1.0 / 4)
    assert out[4].score == pytest.approx(1.0 / 5)


# ---------------------------------------------------------------------------
# query 統合テスト: trigram のみ（degrade）
# ---------------------------------------------------------------------------


@pytest.fixture
def trigram_conn(conn, doc_dir):
    """trigram インデックスのみ構築済みの接続。"""
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
    upsert_document(
        conn,
        collection="test",
        path="c.md",
        body="サーバーの設定と運用について説明します",
        title="サーバー運用ガイド",
        mtime=1002,
    )
    conn.commit()
    set_meta(conn, "trigram_indexed", "1")
    conn.commit()
    return conn


def test_query_trigram_only(trigram_conn):
    """trigram のみ（morph/vec 未構築）で query がヒットする。"""
    results = query(trigram_conn, "形態素解析")
    assert len(results) >= 1
    assert any("a.md" in r.filepath for r in results)


def test_query_returns_search_result(trigram_conn):
    """SearchResult dataclass の全フィールドが設定されている。"""
    results = query(trigram_conn, "形態素解析")
    assert results
    r = results[0]
    assert r.docid
    assert r.filepath
    assert r.score > 0
    assert isinstance(r.snippet, str)
    assert isinstance(r.body, str)
    assert r.body == "形態素解析は日本語の自然言語処理の基礎技術です"


def test_query_n_limit(trigram_conn):
    """n=1 で最大1件を返す。"""
    results = query(trigram_conn, "す", n=1)
    assert len(results) <= 1


def test_query_all_results(trigram_conn):
    """all_results=True で n 制限を超えて全件返す。"""
    results_n1 = query(trigram_conn, "す", n=1)
    results_all = query(trigram_conn, "す", all_results=True)
    assert len(results_all) >= len(results_n1)


def test_query_no_results(trigram_conn):
    """存在しないキーワードは 0 件。"""
    results = query(trigram_conn, "XYZNONEXISTENT999ZZZZZ")
    assert results == []


def test_query_score_ordering(trigram_conn):
    """スコアが降順になっている。"""
    results = query(trigram_conn, "す", n=10, all_results=True)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_collection_filter(conn, tmp_path):
    """collection フィルタが正しく動作する。"""
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

    results = query(conn, "日本語", collection="col1")
    assert all("col1/" in r.filepath for r in results)


def test_query_min_score_filter(trigram_conn):
    """min_score による足切りが機能する。"""
    all_res = query(trigram_conn, "形態素解析", all_results=True)
    if not all_res:
        pytest.skip("検索結果が0件")
    max_score = max(r.score for r in all_res)
    filtered = query(
        trigram_conn, "形態素解析", all_results=True, min_score=max_score + 1.0
    )
    assert filtered == []


# ---------------------------------------------------------------------------
# query 統合テスト: morph 寄与（SudachiPy 必要）
# ---------------------------------------------------------------------------

sudachipy = pytest.importorskip("sudachipy")


def _insert_morph(conn, collection, path, title, body):
    from jaqmd.tokenize.morph import tokenize_text

    upsert_document(
        conn, collection=collection, path=path, body=body, title=title, mtime=1000
    )
    row = conn.execute(
        "SELECT docid FROM documents WHERE collection=? AND path=?",
        (collection, path),
    ).fetchone()
    docid = row["docid"]
    conn.execute(
        "INSERT INTO docs_fts_morph(docid, filepath, title, body) VALUES (?, ?, ?, ?)",
        (
            docid,
            f"{collection}/{path}",
            tokenize_text(title or ""),
            tokenize_text(body),
        ),
    )
    return docid


@pytest.fixture
def hybrid_conn(conn, doc_dir):
    """trigram + morph 両方構築済みの接続。"""
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
    upsert_document(
        conn,
        collection="test",
        path="c.md",
        body="サーバーの設定と運用について説明します",
        title="サーバー運用ガイド",
        mtime=1002,
    )
    conn.commit()
    set_meta(conn, "trigram_indexed", "1")

    # morph FTS に投入
    for path, title, body in [
        (
            "a.md",
            "形態素解析について",
            "形態素解析は日本語の自然言語処理の基礎技術です",
        ),
        ("b.md", "検索エンジン入門", "検索エンジンの仕組みと実装方法を解説します"),
        ("c.md", "サーバー運用ガイド", "サーバーの設定と運用について説明します"),
    ]:
        _insert_morph(conn, "test", path, title, body)
    conn.commit()
    set_meta(conn, "morph_indexed", "1")
    conn.commit()
    return conn


def test_query_with_morph_uses_fusion(hybrid_conn):
    """morph 構築済みの場合、融合結果が返る（複数インデックスが乗る）。"""
    results = query(hybrid_conn, "形態素解析")
    assert len(results) >= 1
    assert any("a.md" in r.filepath for r in results)


def test_query_hybrid_score_ordering(hybrid_conn):
    """RRF 融合後もスコア降順が保たれる。"""
    results = query(hybrid_conn, "す", all_results=True)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_hybrid_no_duplicate_docids(hybrid_conn):
    """RRF 融合後に docid の重複がない。"""
    results = query(hybrid_conn, "す", all_results=True)
    docids = [r.docid for r in results]
    assert len(docids) == len(set(docids))


# ---------------------------------------------------------------------------
# query 統合テスト: Query Expansion 配線
# ---------------------------------------------------------------------------


def test_query_qe_disabled_matches_raw_query(trigram_conn, monkeypatch):
    """qe_enabled=False では qe.expand が呼ばれず raw クエリのみで検索する。"""
    calls = []
    monkeypatch.setattr(
        "jaqmd.search.query.qe_expand",
        lambda *a, **k: calls.append(1) or None,
    )
    query(trigram_conn, "形態素解析", qe_enabled=False)
    assert calls == []


def test_query_qe_none_degrades_to_raw_query(trigram_conn, monkeypatch):
    """qe.expand が None を返す（未導入/失敗）場合、raw クエリの結果と一致する。"""
    monkeypatch.setattr("jaqmd.search.query.qe_expand", lambda *a, **k: None)
    with_qe = query(trigram_conn, "形態素解析", qe_enabled=True)
    without_qe = query(trigram_conn, "形態素解析", qe_enabled=False)
    assert [r.docid for r in with_qe] == [r.docid for r in without_qe]


def test_query_qe_lex_expansion_reaches_trigram(trigram_conn, monkeypatch):
    """lex 展開語が trigram 検索に渡り、raw クエリ単独では拾えない文書もヒットする。"""
    from jaqmd.qe import ExpansionResult

    monkeypatch.setattr(
        "jaqmd.search.query.qe_expand",
        lambda *a, **k: ExpansionResult(
            lex=["サーバー設定"], vec="サーバーの設定について", hyde=""
        ),
    )
    results = query(trigram_conn, "XYZNONEXISTENT999ZZZZZ")
    assert any("c.md" in r.filepath for r in results)


def test_query_qe_vec_expansion_used_for_vsearch(hybrid_conn, monkeypatch):
    """vec 展開文が vsearch に渡されることを確認する（vsearch 呼び出し引数を検証）。"""
    from jaqmd.qe import ExpansionResult
    from jaqmd.store import set_meta

    set_meta(hybrid_conn, "vec_indexed", "1")
    hybrid_conn.commit()

    monkeypatch.setattr(
        "jaqmd.search.query.qe_expand",
        lambda *a, **k: ExpansionResult(
            lex=[], vec="展開されたベクトルクエリ", hyde="仮想文書テキスト"
        ),
    )

    captured_queries = []

    def _fake_vsearch(conn, q, **kwargs):
        captured_queries.append(q)
        return []

    # vsearch は query() 内で `from .vsearch import vsearch` と遅延 import されるため、
    # 呼び出し時に解決される実体側（jaqmd.search.vsearch モジュール）をパッチする。
    monkeypatch.setattr("jaqmd.search.vsearch.vsearch", _fake_vsearch)

    query(hybrid_conn, "元のクエリ")
    assert "展開されたベクトルクエリ" in captured_queries
    assert "仮想文書テキスト" in captured_queries


# ---------------------------------------------------------------------------
# query_searches 統合テスト（tobi/qmd 風 typed searches、MCP query ツール用）
# ---------------------------------------------------------------------------


def test_query_searches_lex_hits(trigram_conn):
    """type=lex の searches で trigram 検索がヒットする。"""
    results = query_searches(trigram_conn, [("lex", "形態素解析")])
    assert len(results) >= 1
    assert any("a.md" in r.filepath for r in results)


def test_query_searches_empty_returns_empty(trigram_conn):
    """searches が空リストなら空を返す。"""
    assert query_searches(trigram_conn, []) == []


def test_query_searches_invalid_type_raises(trigram_conn):
    """type が lex/vec/hyde 以外なら ValueError。"""
    with pytest.raises(ValueError):
        query_searches(trigram_conn, [("bogus", "テスト")])


def test_query_searches_vec_ignored_without_vec_index(trigram_conn):
    """vec_indexed が立っていない場合、type=vec は無視され結果は空（vsearch 未呼び出し）。"""
    results = query_searches(trigram_conn, [("vec", "形態素解析")])
    assert results == []


def test_query_searches_limit(trigram_conn):
    """limit で返却件数が制限される。"""
    results = query_searches(trigram_conn, [("lex", "す")], limit=1)
    assert len(results) <= 1


def test_query_searches_collections_filter(conn, tmp_path):
    """collections（OR リスト）でコレクション絞り込みができる。"""
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

    results = query_searches(conn, [("lex", "日本語")], collections=["col1"])
    assert results
    assert all(r.filepath.startswith("col1/") for r in results)


def test_query_searches_first_search_weighted(hybrid_conn):
    """先頭 search の weight=2.0 により、先頭のみヒットする docid が優先される。

    trigram/morph 両方に "サーバー" を含む c.md がある状況で、
    先頭 lex search と後続 lex search で異なる語を与えたとき、
    先頭側でのみ強くヒットする文書が上位に来ることを確認する。
    """
    results = query_searches(
        hybrid_conn,
        [("lex", "サーバー"), ("lex", "検索エンジン")],
        rerank_enabled=False,
    )
    assert results
    # 両方の語に関連するため c.md（サーバー）/ b.md（検索エンジン）が含まれる
    filepaths = [r.filepath for r in results]
    assert any("c.md" in fp for fp in filepaths)


def test_query_searches_no_rerank(trigram_conn):
    """rerank_enabled=False でも動作する（恒等フォールバック）。"""
    results = query_searches(
        trigram_conn, [("lex", "形態素解析")], rerank_enabled=False
    )
    assert len(results) >= 1
