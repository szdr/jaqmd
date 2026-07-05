from __future__ import annotations


def extract_snippet(text: str, terms: list[str], *, max_chars: int = 200) -> str:
    """テキストから検索語に関連する部分を snippet として抽出する。

    terms の trigram と各文の重なり数でスコアリングし、最高スコアの文を
    中心に max_chars 以内で前後の文を含めて返す。

    Args:
        text:      本文テキスト。
        terms:     検索語トークンのリスト。
        max_chars: 返す snippet の最大文字数（目安）。省略記号を含む場合あり。

    Returns:
        snippet 文字列。text が max_chars 以下の場合はそのまま返す。
    """
    if len(text) <= max_chars:
        return text

    # --- 1. 文に分割 ---
    from ..chunk import _split_sentences

    sentences = _split_sentences(text)
    if not sentences:
        return text[:max_chars]

    # --- 2. terms から trigram セット生成 ---
    from ..tokenize.trigram import _trigrams

    q_trigrams: set[str] = set()
    for tok in terms:
        q_trigrams.update(_trigrams(tok))

    # --- 3. 各文をスコアリング ---
    def _score(sent: str) -> int:
        if q_trigrams:
            return sum(1 for t in q_trigrams if t in sent)
        # trigram が生成できない短い terms はトークン単純一致でフォールバック
        return sum(sent.count(tok) for tok in terms)

    scores = [_score(s) for s, _ in sentences]

    # --- 4. 中心文を選択 ---
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    # 全スコアが 0 なら先頭を使う（フォールバック）
    if scores[best_idx] == 0:
        best_idx = 0

    # --- 5. 中心文から前後に広げる ---
    selected = {best_idx}
    total_chars = len(sentences[best_idx][0])

    lo, hi = best_idx, best_idx
    # 後続・先行を交互に追加
    while True:
        added = False
        # 後方 (+1)
        if hi + 1 < len(sentences):
            cand = sentences[hi + 1][0]
            if total_chars + len(cand) <= max_chars:
                hi += 1
                selected.add(hi)
                total_chars += len(cand)
                added = True
        # 前方 (-1)
        if lo - 1 >= 0:
            cand = sentences[lo - 1][0]
            if total_chars + len(cand) <= max_chars:
                lo -= 1
                selected.add(lo)
                total_chars += len(cand)
                added = True
        if not added:
            break

    # --- 6. 原文順に連結 ---
    snippet = "".join(sentences[i][0] for i in range(lo, hi + 1))

    # --- 7. 省略記号付与 ---
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(sentences) - 1 else ""
    return prefix + snippet + suffix
