# AGENTS.md

このファイルは、jaqmd の開発に携わる AI エージェント（Claude Code など）および開発者向けの実装方針をまとめたものです。

## プロジェクト概要

jaqmd は日本語ドキュメントに特化したローカル検索エンジンです。tobi/qmd からインスピレーションを受けつつ、日本語テキストの検索品質に振り切っています。

**核心となる差別化点:**
- trigram / 形態素解析 / ベクトルの3つの日本語向け検索方式を提供
- Qwen等を一切使わず、ruri（名古屋大学）を採用
- PyTorch非依存の薄い構成

## 基本原則

### 依存関係を薄く保つ

- **PyTorch を引き込まない。** ベクトル推論は fastembed（ONNX Runtime）経由で行う
- 形態素解析・ベクトル検索は**オプション依存**とし、最小構成では trigram 検索のみで動作させる
- 新しい依存を追加する前に、本当に必要か検討する

### モデルの利用方針

- Qwen 系モデルを embedding・rerank・query expansion のいずれにも使用しない
- embedding は `cl-nagoya/ruri-v3-310m`、rerank は `cl-nagoya/ruri-v3-reranker-310m` を使う
- query expansion は独自開発する（`szdr/jaqmd-qe-gemma-4-e2b-it`、Gemma 4 E2B の LoRA fine-tune。GGUF 形式で llama-cpp-python 経由）

### 段階的なインデックス構築を尊重する

- `update`（trigram）は常時、`morph`（形態素）と `embed`（ベクトル）はオプション
- どのインデックスが構築済みかは `index_meta` テーブルで管理する
- 検索コマンドは事前要件を満たさない場合、具体的な次のアクションを提示してエラーにする
  - 例: 「形態素インデックスがありません。`jaqmd morph` を実行してください」

## 技術スタック

| 領域 | 採用技術 |
|------|----------|
| 言語 | Python 3.10+ |
| ストレージ | SQLite + sqlite-vec + FTS5 |
| ベクトル推論 | fastembed（ONNX Runtime） |
| 形態素解析 | SudachiPy（sudachidict-core） |
| Embedding | ruri-v3-310m（ONNX） |
| Reranker | ruri-v3-reranker-310m（ONNX） |
| CLI | Typer または Click |
| MCP | Python MCP SDK |
| パッケージ管理 | uv |

## コマンド体系

```
インデックス構築:
  jaqmd collection add <path> --name <name>
  jaqmd update [--pull]    # trigram FTS（常時）
  jaqmd morph              # 形態素 FTS（オプション）
  jaqmd embed [-f]         # ベクトル（オプション）

検索:
  jaqmd search   <query>   # trigram BM25
  jaqmd mosearch <query>   # 形態素 BM25
  jaqmd vsearch  <query>   # ベクトル
  jaqmd query    <query>   # ハイブリッド + rerank

管理:
  jaqmd get / multi-get / collection / ls / status / cleanup / mcp
```

## ストレージ設計

データベースは `~/.cache/jaqmd/index.sqlite` に保存する（`XDG_CACHE_HOME` を尊重）。

### スキーマ

```sql
-- コンテンツ実体（content-addressable、重複排除）
CREATE TABLE content (
    hash  TEXT PRIMARY KEY,   -- SHA-256
    body  TEXT NOT NULL
);

-- ドキュメント（ファイルパス → ハッシュのマッピング）
CREATE TABLE documents (
    id           INTEGER PRIMARY KEY,
    collection   TEXT NOT NULL,
    path         TEXT NOT NULL,        -- コレクション相対パス
    hash         TEXT NOT NULL REFERENCES content(hash),
    docid        TEXT UNIQUE NOT NULL, -- 6文字ハッシュ（#abc123）
    title        TEXT,
    mtime        INTEGER,
    active       INTEGER DEFAULT 1,    -- 論理削除フラグ
    indexed_at   INTEGER DEFAULT (unixepoch())
);
CREATE INDEX idx_documents_collection ON documents(collection);
CREATE INDEX idx_documents_hash       ON documents(hash);

-- コレクション定義
CREATE TABLE collections (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    path       TEXT NOT NULL,
    glob_mask  TEXT NOT NULL DEFAULT '**/*.md',
    created_at INTEGER DEFAULT (unixepoch())
);

-- パスコンテキスト
CREATE TABLE path_contexts (
    path    TEXT PRIMARY KEY,
    context TEXT NOT NULL
);

-- DBメタ情報（インデックス構築状態・モデル情報）
CREATE TABLE index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- 例: schema_version, morph_indexed, morph_tokenizer,
--     vec_indexed, embed_model, embed_dim

-- FTS1: trigram（trigram tokenizer、辞書不要・デフォルト）
CREATE VIRTUAL TABLE docs_fts_trigram USING fts5(
    docid    UNINDEXED,
    filepath UNINDEXED,    -- collection/path（絞り込み用）
    title,
    body,
    tokenize = 'trigram'
);

-- FTS2: 形態素（unicode61、Python側で分かち書き済みを投入）
CREATE VIRTUAL TABLE docs_fts_morph USING fts5(
    docid    UNINDEXED,
    filepath UNINDEXED,
    title,
    body,
    tokenize = 'unicode61'
);

-- チャンク本体（ベクトル検索用）
CREATE TABLE chunk_vectors (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    docid       TEXT NOT NULL,
    chunk_seq   INTEGER NOT NULL,   -- チャンク番号（0始まり）
    chunk_pos   INTEGER NOT NULL,   -- ドキュメント内の文字オフセット
    chunk_text  TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    UNIQUE(docid, chunk_seq)
);

-- sqlite-vec ベクトルインデックス（ruri-v3-310m は 768次元）
CREATE VIRTUAL TABLE vectors_vec USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding float[768]
);

-- Query Expansion キャッシュ
CREATE TABLE qe_cache (
    query_hash TEXT PRIMARY KEY,
    query_raw  TEXT NOT NULL,
    lex_query  TEXT,
    vec_query  TEXT,
    hyde_text  TEXT,
    model_id   TEXT NOT NULL,
    created_at INTEGER DEFAULT (unixepoch()),
    ttl        INTEGER DEFAULT 86400
);
```

### トリガーによる FTS 同期

trigram FTS は documents の更新に追従してトリガーで自動同期する。

```sql
CREATE TRIGGER documents_ai AFTER INSERT ON documents
WHEN NEW.active = 1 BEGIN
    INSERT INTO docs_fts_trigram(docid, filepath, title, body)
    SELECT NEW.docid,
           NEW.collection || '/' || NEW.path,
           NEW.title,
           c.body
    FROM content c WHERE c.hash = NEW.hash;
END;

CREATE TRIGGER documents_soft_delete AFTER UPDATE ON documents
WHEN OLD.active = 1 AND NEW.active = 0 BEGIN
    DELETE FROM docs_fts_trigram WHERE docid = OLD.docid;
    DELETE FROM docs_fts_morph   WHERE docid = OLD.docid;
END;
```

**形態素 FTS（docs_fts_morph）はトリガーで同期しない。** SQLite トリガー内では SudachiPy を呼べないため、`jaqmd morph` 実行時に Python 側で分かち書きしてから INSERT する。

## 検索パイプライン

### search（trigram）

trigram tokenizer に対してクエリを変換し BM25 で検索する。raw text をそのまま投入する。

日本語のトークン化に関する注意:
- クエリを trigram（3文字）に展開して OR 結合する
  - 例: "東京都庁" → `"東京都" OR "京都庁"`
- ひらがな・カタカナは低情報語が多く、ノイズになりやすい
- 3文字未満のトークンは trigram を生成できないためスキップする

### mosearch（形態素）

SudachiPy で分かち書きしたクエリを unicode61 FTS に対して BM25 検索する。インデックス側もクエリ側も同じ正規化・分かち書きを通すこと（投入時と検索時で処理を揃えないとマッチしない）。

### vsearch（ベクトル）

クエリを ruri-v3 で embedding 化し、vectors_vec を KNN 検索する。

ruri-v3 のプレフィックス仕様を厳守すること:
- クエリ: `検索クエリ: ` を付与
- 文書: `検索文書: ` を付与

これを忘れると検索品質が大きく劣化する。

### query（ハイブリッド）

```
1. Query Expansion で lex: / vec: / hyde: に展開（qe_cache 活用）
2. 利用可能なインデックスで検索:
     - search（常に）
     - mosearch（morph_indexed なら）
     - vsearch（vec_indexed なら）
3. RRF（Reciprocal Rank Fusion, k=60）で融合
4. ruri-reranker で上位候補をリランク
5. 結果を返却
```

利用可能なインデックスは `index_meta` を見て判定する。morph/embed 未実行でも query は動作する。

## fastembed の使い方

embedding・reranker ともにカスタムモデル登録で ruri-v3 を読み込む。

```python
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

TextEmbedding.add_custom_model(
    model="sirasagi62/ruri-v3-310m-ONNX",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="sirasagi62/ruri-v3-310m-ONNX"),
    dim=768,
    model_file="onnx/model.onnx",
)
```

注意点:
- prefix（`検索クエリ:` / `検索文書:`）の付与は fastembed では自動化されないので、embed に渡す前に自前で付ける
- ruri-v3-reranker の ONNX 版が公式に存在しない場合は optimum でエクスポートする（`jaqmd-qe` または別スクリプトで管理）

## チャンク戦略

- 単位: トークン数ベース（ruri-v3 のトークナイザ基準）
- デフォルト: 800トークン、15%オーバーラップ（qmd を踏襲）
- 文境界（。！？）を優先的に分割点とする
- ruri-v3 は 8192 トークンまで対応するため、長めのチャンクも選択肢

## 対応ドキュメント形式（MVP）

- **MVP**: Markdown（`.md`）、プレーンテキスト（`.txt`）
- **将来**: PDF（日本語技術文書が多いため需要は高いが、依存が増えるため後回し）

## 開発上の禁止事項

- **DB を手書き DDL で直接書き換えない。** スキーマ変更は冪等な `schema.sql`（`CREATE ... IF NOT EXISTS` ＋ トリガーは `DROP TRIGGER IF EXISTS` + `CREATE TRIGGER` で再生成）への追記経由で行い、接続時の初期化で自動反映する
- **`jaqmd update` / `morph` / `embed` を自動実行しない。** ユーザーに実行コマンドを提示する
- **PyTorch を依存に追加しない。**
- **形態素 FTS をトリガーで同期しようとしない**（SudachiPy が呼べない）

## ディレクトリ構成（想定）

```
jaqmd/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── src/jaqmd/
│   ├── __init__.py
│   ├── cli.py            # CLI エントリポイント
│   ├── store.py          # SQLite ストレージ層
│   ├── schema.sql        # スキーマ定義
│   ├── tokenize/
│   │   ├── trigram.py    # trigram トークナイズ
│   │   └── morph.py      # SudachiPy ラッパー
│   ├── search/
│   │   ├── trisearch.py
│   │   ├── mosearch.py
│   │   ├── vsearch.py
│   │   └── query.py      # ハイブリッド + RRF + rerank
│   ├── embed.py          # fastembed ラッパー
│   ├── rerank.py         # ruri-reranker ラッパー
│   ├── chunk.py          # チャンク分割
│   └── mcp/
│       └── server.py     # MCP サーバー
└── tests/
```

## テスト方針

- 日本語の表記ゆれ（サーバー/サーバ、第1条/第一条）のマッチを検証する
- trigram と形態素で検索結果が期待通り変わることを確認する
- ruri-v3 のプレフィックス付与漏れを検出するテストを置く
- インデックス未構築時のエラーメッセージが適切であることを確認する

## パッケージ管理メモ

```toml
[project.optional-dependencies]
morph  = ["sudachipy>=0.6", "sudachidict-core"]
vector = ["fastembed>=0.8"]
all    = ["sudachipy>=0.6", "sudachidict-core", "fastembed>=0.8"]

[project.scripts]
jaqmd = "jaqmd.cli:main"
```

最小構成（コア）の依存は sqlite-vec のみに抑える。
