import pytest
from typer.testing import CliRunner
from jaqmd.cli import app

runner = CliRunner()


def test_collection_add(tmp_cache, doc_dir):
    result = runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    assert result.exit_code == 0
    assert "test" in result.output


def test_collection_add_nonexistent(tmp_cache):
    result = runner.invoke(app, ["collection", "add", "/no/such/path/xyz", "--name", "test"])
    assert result.exit_code != 0


def test_collection_list(tmp_cache, doc_dir):
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    result = runner.invoke(app, ["collection", "list"])
    assert result.exit_code == 0
    assert "test" in result.output


def test_collection_remove(tmp_cache, doc_dir):
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    result = runner.invoke(app, ["collection", "remove", "test"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["collection", "list"])
    assert "test" not in result.output


def test_update_empty_collection(tmp_cache, doc_dir):
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "完了" in result.output


def test_update_with_files(tmp_cache, doc_dir):
    (doc_dir / "a.md").write_text("# 形態素解析\n日本語の自然言語処理について説明します。")
    (doc_dir / "b.md").write_text("# 検索エンジン\n検索エンジンの仕組みを解説します。")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0


def test_search_without_index(tmp_cache):
    result = runner.invoke(app, ["search", "テスト"])
    assert result.exit_code != 0
    assert "update" in result.output


def test_search_with_results(tmp_cache, doc_dir):
    (doc_dir / "a.md").write_text("# 形態素解析\n日本語の形態素解析の詳細な解説です。")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    runner.invoke(app, ["update"])
    result = runner.invoke(app, ["search", "形態素解析"])
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_search_no_results(tmp_cache, doc_dir):
    (doc_dir / "a.md").write_text("# テスト\n検索テスト文書です。")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    runner.invoke(app, ["update"])
    result = runner.invoke(app, ["search", "絶対存在しないXYZ999"])
    assert result.exit_code == 0
    assert "No results" in result.output


def test_search_json_output(tmp_cache, doc_dir):
    import json
    (doc_dir / "a.md").write_text("# 形態素解析\n日本語処理の基礎技術について詳しく解説します。")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    runner.invoke(app, ["update"])
    result = runner.invoke(app, ["search", "形態素解析", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_get_command(tmp_cache, doc_dir):
    (doc_dir / "a.md").write_text("# 形態素解析\n本文の内容です。")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    runner.invoke(app, ["update"])

    from jaqmd.store import connect
    conn = connect()
    row = conn.execute("SELECT docid FROM documents WHERE active=1 LIMIT 1").fetchone()
    docid = row["docid"]

    result = runner.invoke(app, ["get", docid])
    assert result.exit_code == 0
    assert "本文の内容" in result.output


def test_get_nonexistent(tmp_cache):
    result = runner.invoke(app, ["get", "xxxxxx"])
    assert result.exit_code != 0


def test_ls_command(tmp_cache, doc_dir):
    (doc_dir / "a.md").write_text("# A\n内容A")
    (doc_dir / "b.md").write_text("# B\n内容B")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    runner.invoke(app, ["update"])
    result = runner.invoke(app, ["ls"])
    assert result.exit_code == 0
    assert "a.md" in result.output
    assert "b.md" in result.output


def test_status_command(tmp_cache, doc_dir):
    (doc_dir / "a.md").write_text("# テスト\n内容")
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    runner.invoke(app, ["update"])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "trigram" in result.output.lower()
    assert "search" in result.output


def test_cleanup_command(tmp_cache, doc_dir):
    runner.invoke(app, ["collection", "add", str(doc_dir), "--name", "test"])
    result = runner.invoke(app, ["cleanup"])
    assert result.exit_code == 0
    assert "完了" in result.output


def test_morph_unimplemented(tmp_cache):
    result = runner.invoke(app, ["morph"])
    assert result.exit_code != 0
    assert "未実装" in result.output


def test_embed_unimplemented(tmp_cache):
    result = runner.invoke(app, ["embed"])
    assert result.exit_code != 0
    assert "未実装" in result.output


def test_mosearch_unimplemented(tmp_cache):
    result = runner.invoke(app, ["mosearch", "テスト"])
    assert result.exit_code != 0
    assert "morph" in result.output


def test_vsearch_unimplemented(tmp_cache):
    result = runner.invoke(app, ["vsearch", "テスト"])
    assert result.exit_code != 0
    assert "embed" in result.output


def test_query_unimplemented(tmp_cache):
    result = runner.invoke(app, ["query", "テスト"])
    assert result.exit_code != 0
    assert "search" in result.output


def test_mcp_unimplemented(tmp_cache):
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code != 0
