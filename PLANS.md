# jaqmd 今後の作業手順（ロードマップ + 作業単位分解）

## Context

`AGENTS.md` は jaqmd の最終形（trigram / 形態素 / ベクトル / ハイブリッド + MCP）を詳細に設計した決定版ドキュメント。一方、現状の実装は **trigram 検索のみ完成**で、`morph` / `embed` / `mosearch` / `vsearch` / `query` / `mcp` は `cli.py:359-431` の「exit 1 で案内するスタブ」のまま。

本計画は AGENTS.md の設計に沿って残りの機能を実装するためのロードマップを定め、各フェーズを**独立して着手・レビュー可能な作業単位**に分解する。各作業単位は「スキーマ → ロジック層 → CLI 配線 → テスト」を1セットとし、既存パターン（`store.py` のDB関数、`schema.sql` のトリガー同期、`search/trisearch.py` の `SearchResult`、`format.py`、`index_meta` による段階的インデックス管理）を最大限再利用する。

各 Phase は後から1つずつ個別に実行する想定。

AGENTS.md の禁止事項を厳守する: PyTorch を入れない / DB を手書き DDL で直接書き換えず冪等な `schema.sql` 追記経由 / 形態素FTSをトリガー同期しない / `update`・`morph`・`embed` を自動実行しない。

---

## 全体ロードマップ

```
Phase 0  スキーマ・マイグレーション基盤   ← 全フェーズの前提
Phase A  形態素検索 (morph / mosearch)    ← SudachiPy のみ、ONNX 不要で着手しやすい
Phase B  ベクトル検索 (embed / vsearch)   ← fastembed(ONNX) + ruri-v3 + sqlite-vec
Phase C  ハイブリッド検索 (query)          ← RRF 融合 + ruri-reranker
Phase D  MCP サーバー (mcp)                ← 検索機能を MCP ツールとして公開
```

依存順: 0 → A → B → C。D は A/B 完成後ならいつでも可（C があると望ましい）。

---

## Phase 0: 冪等スキーマ初期化基盤（前提）

**方針:** マイグレーション機構は採用しない。tobi/qmd の `initializeDatabase()` 方式を採用し、接続のたびに `schema.sql` を無条件実行することで、後フェーズで追記した新テーブルが既存DBへ自動反映される設計にする。

### 作業単位 0-1: 冪等初期化への移行
- `schema.sql`: 全 `CREATE TABLE`/`CREATE INDEX`/`CREATE VIRTUAL TABLE` を `IF NOT EXISTS` 付きに変更。全トリガーを `DROP TRIGGER IF EXISTS <name>;` + `CREATE TRIGGER <name> ...` 形式に変更（Phase A でトリガー本体に morph DELETE 行を追記したとき、既存DBへ確実に反映するため）。
- `store.py:_ensure_schema()`: `index_meta` 存在チェックと `schema_version` INSERT を削除し、毎回 `executescript(schema.sql)` を実行する形に簡略化。`connect()` から呼ぶ構造は維持。`schema_version` の概念は廃止（`index_meta` テーブルと `get_meta`/`set_meta` は `morph_indexed` 等で引き続き使用するため存続）。
- `src/jaqmd/migrations/` は**新設しない**。
- **vec0 例外**: `vectors_vec`（`USING vec0`）は sqlite-vec 拡張ロード後でないと CREATE が失敗するため `schema.sql` には含めない。Phase B で `connect()` 内の拡張ロード成功時のみ `CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec ...` を別途実行する。
- テスト: 旧バージョンDB（trigram のみ・`docs_fts_morph` なし）に接続 → 新テーブルが自動作成されることと、トリガー再生成が冪等であることを `test_store.py` に追加。

---

## Phase A: 形態素検索（morph / mosearch）

ONNX 不要・追加依存は `sudachipy` + `sudachidict-core`（`pyproject.toml` の `[morph]` extras に既存）。

### 作業単位 A-1: morph FTS スキーマ
- `schema.sql` に `docs_fts_morph`（`CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts_morph USING fts5(...)`, `tokenize='unicode61'`, カラムは trigram 版と同構成）を追記。
- **同期方針（AGENTS.md 厳守）**: INSERT はトリガーで同期しない（SudachiPy をトリガー内で呼べないため）。一方 **DELETE 系は安全**なので、`documents_soft_delete` / `documents_au_hash` / `documents_delete` トリガー（`schema.sql` 内）に `docs_fts_morph` からの削除を追記する。Phase 0 で導入した DROP+CREATE 方式により、接続時に既存DBへ自動反映される。INSERT は `jaqmd morph` 実行時に Python で行う。

### 作業単位 A-2: 形態素トークナイザ
- `src/jaqmd/tokenize/morph.py` を新設。SudachiPy ラッパー。`tokenize/trigram.py` と同じく「分かち書き関数 + FTS クエリ変換関数」の2関数構成にする。
- **インデックス側とクエリ側で同一の正規化・分かち書き**を通すこと（AGENTS.md mosearch 節）。正規化形（normalized_form）の採用可否をここで決める。
- `sudachipy` は optional import（未インストール時は明確なエラーで `pip install 'jaqmd[morph]'` を案内）。Tokenizer はモジュールレベルでキャッシュ（生成コスト大）。

### 作業単位 A-3: `jaqmd morph` コマンド実装
- `cli.py:359-367` のスタブを置換。`update` コマンド（`cli.py:84-134`）の構造を踏襲: 全 active ドキュメントの本文を取得 → `morph.py` で分かち書き → `docs_fts_morph` に INSERT。
- 完了後 `set_meta(conn, "morph_indexed", "1")`、`set_meta(conn, "morph_tokenizer", ...)` を記録（`store.py:173` の既存関数を再利用）。
- 再実行時は `docs_fts_morph` を作り直す（DELETE → 再投入）冪等な実装にする。

### 作業単位 A-4: mosearch 検索ロジック + CLI 配線
- `src/jaqmd/search/mosearch.py` を新設。`trisearch.py` をテンプレートに、クエリを `morph.py` で分かち書き → `docs_fts_morph MATCH` で BM25。**戻り値は既存 `SearchResult` dataclass を再利用**（`format.py` がそのまま使える）。
- `cli.py:381-392` の `mosearch` スタブを置換。`_run_search`（`cli.py:141-178`）と同等の出力オプション群（`-n/-c/--min-score/--all/--full/--json/--md/--xml/--files`）を持たせる。事前要件チェックは `get_meta(conn, "morph_indexed")`、未構築なら「`jaqmd morph` を実行してください」と案内（既存 `_run_search` のエラーパターン踏襲）。
- `_run_search` を検索バックエンド差し替え可能に一般化して mosearch/vsearch と共有するのが望ましい。

### 作業単位 A-5: テスト
- `test_morph_tokenize.py`: 分かち書き・正規化の単体テスト。
- `test_mosearch.py`: 表記ゆれ（サーバー/サーバ、第1条/第一条）が trigram と異なる挙動でマッチすること（AGENTS.md テスト方針）。
- `test_cli.py` 拡張: `morph` 実行 → `mosearch` 成功、未実行時の案内エラー。
- SudachiPy 未導入CI を考慮し `pytest.importorskip("sudachipy")` でガード。

---

## Phase B: ベクトル検索（embed / vsearch）

追加依存は `fastembed`（`[vector]` extras に既存）。`sqlite-vec` はコア依存に既存だが**未ロード**——`store.connect()` で拡張ロードが必要。

### 作業単位 B-1: ベクトルスキーマ + sqlite-vec ロード
- `schema.sql` に `chunk_vectors`（AGENTS.md スキーマ、`CREATE TABLE IF NOT EXISTS`）を追記。
- `vectors_vec`（`vec0`）は **Phase 0 の vec0 例外**に従い `schema.sql` には含めない。`store.py:connect()` で sqlite-vec 拡張ロード後に `CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(embedding float[768])` を実行する。
- `store.py:13-19` `connect()` に `import sqlite_vec; conn.enable_load_extension(True); sqlite_vec.load(conn)` を追加（vec0 仮想テーブル利用に必須）。拡張を使えない環境向けにエラーハンドリング。

### 作業単位 B-2: チャンク分割
- `src/jaqmd/chunk.py` を新設。AGENTS.md チャンク戦略: 800トークン・15%オーバーラップ・文境界（。！？）優先。トークン基準は ruri-v3 トークナイザ。
- 戻り値は `(chunk_seq, chunk_pos, chunk_text)` のリスト。純粋関数として単体テスト可能に。

### 作業単位 B-3: embedding ラッパー
- `src/jaqmd/embed.py` を新設。AGENTS.md「fastembed の使い方」節の `TextEmbedding.add_custom_model(...)`（`cl-nagoya/ruri-v3-310m`, MEAN, 768次元）をそのまま使用。
- **prefix を自前付与**（fastembed は自動化しない）: 文書 `検索文書: ` / クエリ `検索クエリ: `。付与漏れ検出テストを置く（AGENTS.md テスト方針）。
- モデルは `~/.cache/jaqmd/models/` にDL。`fastembed` は optional import。

### 作業単位 B-4: `jaqmd embed` コマンド実装
- `cli.py:370-378` のスタブを置換。active ドキュメント本文 → `chunk.py` で分割 → `embed.py` で文書 prefix 付きベクトル化 → `chunk_vectors` + `vectors_vec` に投入。
- `-f` フラグ（既存ベクトルの再生成）を AGENTS.md コマンド体系に合わせて実装。
- 完了後 `set_meta` で `vec_indexed=1`, `embed_model`, `embed_dim=768` を記録。

### 作業単位 B-5: vsearch 検索ロジック + CLI 配線
- `src/jaqmd/search/vsearch.py` を新設。クエリに `検索クエリ: ` を付け embedding → `vectors_vec` を KNN（`vec0` の `MATCH`/`k`）→ `chunk_vectors` 経由で docid 解決 → `SearchResult` に整形（chunk_text をスニペットに）。
- `cli.py:395-405` の `vsearch` スタブを置換。事前要件は `vec_indexed`、未構築なら `jaqmd embed` を案内。
- `status`（`cli.py:293-336`）と `get_stats`（`store.py:181-197`）は既に morph/vec を考慮済みなので、メタが立てば自動で ✓ 表示される。

### 作業単位 B-6: テスト
- `test_chunk.py`: 分割境界・オーバーラップの単体テスト（モデル不要）。
- `test_embed.py`: prefix 付与の検証（embedding 本体はモック or `importorskip`）。
- ベクトル検索の統合テストは重いので `@pytest.mark.integration` で分離し、CI 既定はスキップ。

---

## Phase C: ハイブリッド検索（query）

### 作業単位 C-1: RRF 融合 + query ロジック
- `src/jaqmd/search/query.py` を新設。AGENTS.md「query」パイプライン: 利用可能インデックス（`index_meta` 判定）で search / mosearch / vsearch を実行 → **RRF（k=60）でランク融合** → 上位候補を返す。morph/vec 未構築でも search のみで動作する degradation を担保。
- 既存の `trisearch` / `mosearch` / `vsearch` を呼び出して結果を集約（再実装しない）。

### 作業単位 C-2: ruri-reranker ラッパー
- `src/jaqmd/rerank.py` を新設。`cl-nagoya/ruri-v3-reranker-310m`。AGENTS.md 注記: reranker の ONNX 版が公式に無ければ optimum でエクスポート（別スクリプト/`jaqmd-qe` 管理）。ONNX 入手手段の確定がこの作業単位のリスク要因——着手時に要確認。
- reranker 未利用時は RRF 結果をそのまま返すフォールバックを用意。

### 作業単位 C-3: Query Expansion（最小実装 + キャッシュ）
- `schema.sql` に `qe_cache` テーブルを追記（`CREATE TABLE IF NOT EXISTS`）。接続時に自動作成される。
- QE 本体は別リポジトリ `jaqmd-qe` 想定（Qwen 禁止・独自開発）。本フェーズでは**QE なし（raw クエリをそのまま使う）でも query が成立する**最小実装に留め、`qe_cache` 配線は枠だけ用意。フル QE は別タスクに切り出す。

### 作業単位 C-4: `jaqmd query` CLI + テスト
- `cli.py:408-418` のスタブを置換。出力は既存 `format.py` 再利用。
- テスト: 各インデックス構築状態の組み合わせで適切に degrade すること、RRF 順位の妥当性。

---

## Phase D: MCP サーバー（mcp）

### 作業単位 D-1: MCP サーバー実装 + CLI
- `src/jaqmd/mcp/server.py` を新設。Python MCP SDK で `search` / `mosearch` / `vsearch` / `query` / `get` を MCP ツールとして公開（既存の検索・取得関数を薄くラップ）。
- `pyproject.toml` に MCP SDK 依存を追加（optional extras 推奨）。
- `cli.py:421-430` の `mcp` スタブを置換（`--http` モード対応）。
- テスト: ツール登録とハンドラの単体テスト。

---

## 横断的な注意点

- **`_run_search` の一般化**: Phase A 時点で `cli.py:141-178` を「検索関数を引数で受ける」形にリファクタし、mosearch/vsearch/query で共有する（重複配線の回避）。
- **`SearchResult` を全検索層で統一**: mosearch/vsearch/query も `search/trisearch.py` の dataclass を返し、`format.py` をそのまま使う。
- **optional 依存のガード**: morph/vector/MCP の各 import は遅延 import + 未導入時の明確な案内メッセージ（既存スタブの案内文体に合わせる）。
- **README 更新**: 各 Phase 完了時に該当コマンドの「未実装」記述を実装済みに更新。

---

## 検証方法（各 Phase 共通）

```bash
# 単体・統合テスト
uv run pytest
uv run pytest -m "not integration"   # モデル無し環境

# 手動 E2E（Phase A 例）
uv run jaqmd collection add ./docs --name docs
uv run jaqmd update
uv run jaqmd morph
uv run jaqmd mosearch "サーバ設定"   # 表記ゆれが trigram と異なる挙動か確認
uv run jaqmd status                  # morph FTS が ✓ になるか
```

- 各作業単位は「テスト緑 + 手動 E2E で当該コマンド成功 + `status` 表示更新」を完了条件とする。Phase 0 では「旧DB（trigram のみ）への再接続で新テーブルが自動作成される」ことをテストで確認する。
- AGENTS.md 禁止事項の自己チェック（PyTorch 非依存、`schema.sql` 追記以外のスキーマ変更なし、トリガーで形態素同期していない）を Phase A/B 完了時に実施。
