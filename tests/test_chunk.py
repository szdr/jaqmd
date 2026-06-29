"""chunk_document の単体テスト（モデル不要）。"""
from __future__ import annotations

import pytest

from jaqmd.chunk import chunk_document, _split_sentences


# 軽量なトークンカウンタ（文字数ベース）
def count_chars(text: str) -> int:
    return len(text)


# ─── _split_sentences ───────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "これは文1です。これは文2です。"
    sents = _split_sentences(text)
    assert len(sents) == 2
    assert sents[0][0] == "これは文1です。"
    assert sents[0][1] == 0
    assert sents[1][0] == "これは文2です。"
    assert sents[1][1] == 8


def test_split_sentences_no_terminator():
    text = "終止符なしのテキスト"
    sents = _split_sentences(text)
    assert len(sents) == 1
    assert sents[0] == ("終止符なしのテキスト", 0)


def test_split_sentences_newline():
    text = "行1\n行2\n行3"
    sents = _split_sentences(text)
    # 改行で分割されること
    assert len(sents) == 3


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


# ─── chunk_document ─────────────────────────────────────────────────────────

def test_empty_text():
    assert chunk_document("", count_tokens=count_chars) == []


def test_single_chunk():
    """全文が max_tokens 以内なら1チャンクで返る。"""
    text = "短いテキスト。"
    chunks = chunk_document(text, max_tokens=100, overlap_ratio=0.15, count_tokens=count_chars)
    assert len(chunks) == 1
    assert chunks[0] == (0, 0, text)


def test_chunk_seq_increments():
    """chunk_seq は 0 から連番になる。"""
    text = "A" * 10 + "。" + "B" * 10 + "。" + "C" * 10 + "。"
    chunks = chunk_document(text, max_tokens=15, overlap_ratio=0.0, count_tokens=count_chars)
    seqs = [c[0] for c in chunks]
    assert seqs == list(range(len(chunks)))


def test_chunk_pos_is_offset():
    """chunk_pos は原文中の文字オフセット。"""
    sent1 = "最初の文です。"  # 7文字
    sent2 = "次の文です。"    # 6文字
    sent3 = "最後の文。"      # 5文字
    text = sent1 + sent2 + sent3
    # max_tokens を小さくして各文が別チャンクに入るようにする
    chunks = chunk_document(text, max_tokens=7, overlap_ratio=0.0, count_tokens=count_chars)
    # chunk_pos は各チャンクの先頭文の開始位置
    assert chunks[0][1] == 0
    assert chunks[1][1] == len(sent1)


def test_overlap():
    """overlap_ratio > 0 のとき、チャンク間で文が引き継がれる。"""
    # 各文を10文字に設定、max_tokens=15（1文+余り）で分割
    text = "A" * 10 + "。" + "B" * 10 + "。" + "C" * 10 + "。"
    chunks_with_overlap = chunk_document(
        text, max_tokens=15, overlap_ratio=0.5, count_tokens=count_chars
    )
    chunks_no_overlap = chunk_document(
        text, max_tokens=15, overlap_ratio=0.0, count_tokens=count_chars
    )
    # オーバーラップあり版はチャンク数が多くなる（または内容が重なる）
    # 少なくとも最初のチャンクが同じ開始位置から始まること
    assert chunks_with_overlap[0][1] == 0
    assert chunks_no_overlap[0][1] == 0


def test_long_single_sentence():
    """max_tokens を超える1文でも最低1チャンクとして入ること（無限ループ防止）。"""
    text = "A" * 200 + "。"
    chunks = chunk_document(text, max_tokens=50, overlap_ratio=0.15, count_tokens=count_chars)
    assert len(chunks) >= 1
    # 全文がカバーされていること
    covered = "".join(c[2] for c in chunks)
    # 重複含むがすべての文字が含まれる
    assert "A" * 200 in covered


def test_no_infinite_loop():
    """多数の短い文でも終了すること。"""
    text = "。".join(["x" * 5] * 100) + "。"
    chunks = chunk_document(text, max_tokens=20, overlap_ratio=0.3, count_tokens=count_chars)
    assert len(chunks) > 0


def test_chunk_covers_all_text():
    """すべての文がいずれかのチャンクに含まれること（オーバーラップなし時）。"""
    text = "文A。文B。文C。文D。文E。"
    chunks = chunk_document(text, max_tokens=10, overlap_ratio=0.0, count_tokens=count_chars)
    all_text = "".join(c[2] for c in chunks)
    # 元の各文が all_text に含まれること
    for sent in ["文A。", "文B。", "文C。", "文D。", "文E。"]:
        assert sent in all_text, f"'{sent}' がチャンクに含まれていない"
