def to_fts_query(query: str) -> str:
    """クエリ文字列を trigram tokenizer 向け FTS5 MATCH 式に変換する。

    trigram は 3 文字以上が必要。2 文字以下はプレフィックス検索にフォールバック。
    """
    query = query.strip()
    if not query:
        return ""
    escaped = query.replace('"', '""')
    if len(query) >= 3:
        return f'"{escaped}"'
    return f'"{escaped}"*'
