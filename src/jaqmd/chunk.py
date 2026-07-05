from __future__ import annotations

import re
from typing import Callable


def chunk_document(
    text: str,
    *,
    max_tokens: int = 800,
    overlap_ratio: float = 0.15,
    count_tokens: Callable[[str], int],
) -> list[tuple[int, int, str]]:
    """テキストをチャンクに分割する。

    文境界（。！？および改行）で分割し、max_tokens を超えたらチャンクを確定する。
    次チャンクは末尾から overlap_ratio 分のトークンを占める文を引き継ぐ。

    Args:
        text: 分割対象テキスト。
        max_tokens: チャンクの最大トークン数。
        overlap_ratio: 隣接チャンク間のオーバーラップ率（トークン数比）。
        count_tokens: テキストのトークン数を返す関数（依存性注入）。

    Returns:
        [(chunk_seq, chunk_pos, chunk_text), ...] のリスト。
        chunk_pos は原文中の文字オフセット。
    """
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # 各文のトークン数は貪欲詰め・オーバーラップ計算で複数回参照されるため、
    # 文ごとに1回だけトークナイズして使い回す。
    sent_tokens = [count_tokens(s) for s, _ in sentences]

    overlap_tokens = max(1, int(max_tokens * overlap_ratio))
    chunks: list[tuple[int, int, str]] = []
    i = 0

    while i < len(sentences):
        # チャンクに文を貪欲に詰める
        sents: list[tuple[str, int]] = []
        total = 0
        j = i
        while j < len(sentences):
            s, pos = sentences[j]
            t = sent_tokens[j]
            if total > 0 and total + t > max_tokens:
                break
            sents.append((s, pos))
            total += t
            j += 1

        chunk_text = "".join(s for s, _ in sents)
        chunk_pos = sents[0][1]
        chunks.append((len(chunks), chunk_pos, chunk_text))

        if j >= len(sentences):
            break

        # 次チャンクの開始位置: 末尾から overlap_tokens 分の文を引き継ぐ
        # j-1 から逆順に文を積み上げ、overlap_tokens を超えたら止まる
        overlap_start = j  # デフォルトは overlap なし
        accumulated = 0
        for k in range(j - 1, i, -1):
            accumulated += sent_tokens[k]
            overlap_start = k
            if accumulated >= overlap_tokens:
                break

        # i が進まない場合は強制的に1文進める（無限ループ防止）
        i = max(overlap_start, i + 1)

    return chunks


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """句点・感嘆符・疑問符・改行で文に分割し、(sentence, start_pos) を返す。

    区切り文字は直前の文に含める（右分割）。
    """
    pattern = re.compile(r"[。！？\n]+")
    result: list[tuple[str, int]] = []
    pos = 0
    for m in pattern.finditer(text):
        end = m.end()
        sent = text[pos:end]
        if sent.strip():
            result.append((sent, pos))
        pos = end
    # 末尾の残り（句点なし）
    if pos < len(text):
        remaining = text[pos:]
        if remaining.strip():
            result.append((remaining, pos))
    return result
