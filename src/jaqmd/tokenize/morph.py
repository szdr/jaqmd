from __future__ import annotations

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            import sudachipy
        except ImportError as e:
            raise ImportError(
                "sudachipy が見つかりません。\n"
                "→ pip install 'jaqmd[morph]' を実行してください。"
            ) from e
        _tokenizer = sudachipy.Dictionary().create()
    return _tokenizer


def tokenize_text(text: str) -> str:
    """テキストを形態素解析して分かち書き文字列（正規化形）を返す。インデックス格納用。"""
    tokenizer = _get_tokenizer()
    return " ".join(m.normalized_form() for m in tokenizer.tokenize(text))


def to_fts_query(query: str) -> str:
    """クエリを形態素解析して FTS5 向け MATCH 式に変換する。重複排除・OR 結合。"""
    query = query.strip()
    if not query:
        return ""
    tokenizer = _get_tokenizer()
    seen: set[str] = set()
    parts: list[str] = []
    for m in tokenizer.tokenize(query):
        norm = m.normalized_form()
        if norm not in seen:
            seen.add(norm)
            escaped = norm.replace('"', '""')
            parts.append(f'"{escaped}"')
    return " OR ".join(parts)


def snippet_terms(query: str) -> list[str]:
    """クエリを形態素解析し、surface 形と正規化形の両方を重複排除して返す。

    原文スニペット生成時の探索候補として使用する。
    """
    query = query.strip()
    if not query:
        return []
    tokenizer = _get_tokenizer()
    seen: set[str] = set()
    terms: list[str] = []
    for m in tokenizer.tokenize(query):
        for form in (m.surface(), m.normalized_form()):
            if form and form not in seen:
                seen.add(form)
                terms.append(form)
    return terms
