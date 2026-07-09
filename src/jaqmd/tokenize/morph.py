from __future__ import annotations

_tokenizer = None

# SudachiPy は入力が約49149バイト（UTF-8）を超えると
# "Tokenization error: Input is too long" を送出するため、
# 余裕を持たせた安全上限。
_MAX_BYTES = 40000


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


def _split_long_line(line: str) -> list[str]:
    """改行を含まない1行が上限バイト数を超える場合に、文字単位で分割する。

    マルチバイト文字の境界を壊さないよう、1文字ずつバイト数を積算する。
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_bytes = 0
    for ch in line:
        ch_bytes = len(ch.encode("utf-8"))
        if buf and buf_bytes + ch_bytes > _MAX_BYTES:
            chunks.append("".join(buf))
            buf = []
            buf_bytes = 0
        buf.append(ch)
        buf_bytes += ch_bytes
    if buf:
        chunks.append("".join(buf))
    return chunks


def _split_for_sudachi(text: str) -> list[str]:
    """SudachiPy の入力バイト長制約を回避するため、改行境界でテキストを分割する。

    バイト予算 _MAX_BYTES を超えない範囲で複数行をまとめてチャンク化する。
    単一行が予算を超える場合は文字単位でさらに分割する。
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_bytes = 0

    def flush() -> None:
        nonlocal buf, buf_bytes
        if buf:
            chunks.append("".join(buf))
            buf = []
            buf_bytes = 0

    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > _MAX_BYTES:
            flush()
            chunks.extend(_split_long_line(line))
            continue
        if buf and buf_bytes + line_bytes > _MAX_BYTES:
            flush()
        buf.append(line)
        buf_bytes += line_bytes
    flush()

    return chunks


def tokenize_text(text: str) -> str:
    """テキストを形態素解析して分かち書き文字列（正規化形）を返す。インデックス格納用。

    SudachiPy の入力バイト長制約を回避するため、内部で改行境界に分割してから
    形態素解析し、結果を連結する。
    """
    tokenizer = _get_tokenizer()
    parts: list[str] = []
    for chunk in _split_for_sudachi(text):
        parts.extend(m.normalized_form() for m in tokenizer.tokenize(chunk))
    return " ".join(parts)


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
