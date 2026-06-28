def _trigrams(token: str) -> list[str]:
    """トークンを trigram（3文字スライディングウィンドウ）に分解する。

    3文字未満のトークンは trigram を生成できないため空リストを返す。
    """
    if len(token) < 3:
        return []
    return [token[i : i + 3] for i in range(len(token) - 2)]


def to_fts_query(query: str) -> str:
    """クエリ文字列を trigram FTS5 向け MATCH 式に変換する。

    各語を trigram（3文字）に分解し、全 trigram を OR で結合する。
    クォートにより FTS5 構文（OR/AND/* など）の誤解釈とインジェクションを防ぐ。
    3文字未満のトークンは trigram を生成できないためスキップする。
    重複する trigram は除去して順序を維持する。

    例: "ABCDE"    → '"ABC" OR "BCD" OR "CDE"'
        "東京都庁" → '"東京都" OR "京都庁" OR "都庁"'  (※4文字→3 trigram)
        "形態素 解析" → '"形態素" OR "態素解" OR "素解析" OR "解析"'  (2トークン連結)
        "AB"       → ""  (3文字未満はスキップ)
    """
    query = query.strip()
    if not query:
        return ""

    seen: set[str] = set()
    parts: list[str] = []
    for token in query.split():
        for tri in _trigrams(token):
            if tri not in seen:
                seen.add(tri)
                escaped = tri.replace('"', '""')
                parts.append(f'"{escaped}"')

    return " OR ".join(parts)
