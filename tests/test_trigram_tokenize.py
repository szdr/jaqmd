import pytest
from jaqmd.tokenize.trigram import _trigrams, to_fts_query


class TestTrigrams:
    def test_basic(self):
        assert _trigrams("ABCDE") == ["ABC", "BCD", "CDE"]

    def test_exactly_3_chars(self):
        assert _trigrams("ABC") == ["ABC"]

    def test_too_short(self):
        assert _trigrams("AB") == []
        assert _trigrams("A") == []
        assert _trigrams("") == []

    def test_japanese(self):
        # 4文字 → range(4-2)=2 trigram
        assert _trigrams("東京都庁") == ["東京都", "京都庁"]

    def test_duplicates_preserved(self):
        # _trigrams 自体は重複を保持する（除去は to_fts_query で行う）
        assert _trigrams("あああ") == ["あああ"]


class TestToFtsQuery:
    def test_abcde(self):
        assert to_fts_query("ABCDE") == '"ABC" OR "BCD" OR "CDE"'

    def test_japanese_4chars(self):
        # 4文字 → 2 trigram
        assert to_fts_query("東京都庁") == '"東京都" OR "京都庁"'

    def test_short_token_skipped(self):
        assert to_fts_query("AB") == ""
        assert to_fts_query("A") == ""

    def test_empty(self):
        assert to_fts_query("") == ""
        assert to_fts_query("   ") == ""

    def test_multiple_tokens(self):
        # "形態素"（3文字→1 trigram）と "解析"（2文字→スキップ）
        result = to_fts_query("形態素 解析")
        assert result == '"形態素"'

    def test_multiple_tokens_all_long(self):
        # 両トークンが3文字以上
        result = to_fts_query("形態素 解析す")
        assert result == '"形態素" OR "解析す"'

    def test_duplicate_trigrams_removed(self):
        # 同じ trigram が複数トークンから出ても1つだけ
        result = to_fts_query("ABCabc ABC")
        parts = result.split(" OR ")
        assert len(parts) == len(set(parts))

    def test_quote_escape(self):
        # ダブルクォートを含む trigram は "" にエスケープされる
        result = to_fts_query('A"BCDE')
        # 'A"B' → 'A""B' としてクォートされる
        assert '""' in result

    def test_order_preserved(self):
        # 最初に登場した trigram の順序が維持される
        result = to_fts_query("ABCDE")
        assert result == '"ABC" OR "BCD" OR "CDE"'
