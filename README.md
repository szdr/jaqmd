# jaqmd

日本語ドキュメントに特化したローカル検索エンジン。

[tobi/qmd](https://github.com/tobi/qmd) からインスピレーションを受けつつ、日本語テキストの検索品質に振り切った設計を採用しています。

## 特徴

- **日本語特化**: trigram検索・形態素解析検索・ベクトル検索の3つを日本語向けに最適化
- **完全ローカル**: SQLite + sqlite-vec + FTS5 で完結。外部DBやサーバー不要
- **段階的なインデックス構築**: 必要な検索機能だけを段階的に有効化できる

## アーキテクチャ概要

```
ドキュメント ──► jaqmd update ──► trigram FTS（常時）
                 jaqmd morph  ──► 形態素 FTS（オプション）
                 jaqmd embed  ──► ベクトル（オプション）

検索:
  jaqmd search    trigram BM25検索
  jaqmd mosearch  形態素解析 BM25検索
  jaqmd vsearch   ベクトル意味検索（ruri-v3）
  jaqmd query     上記を組み合わせ + リランク
```

## モデル構成

| 役割 | モデル | 提供元 | ライセンス |
|------|--------|--------|-----------|
| Embedding | `cl-nagoya/ruri-v3-310m` | 名古屋大学 | Apache-2.0 |
| Reranker | `cl-nagoya/ruri-v3-reranker-310m` | 名古屋大学 | Apache-2.0 |
| Query Expansion | `szdr/jaqmd-qe-gemma-4-e2b-it`（Gemma 4 E2B LoRA） | jaqmd | - |

Embedding・Reranker は ONNX 形式で fastembed 経由、Query Expansion は GGUF 形式で llama-cpp-python 経由で読み込みます。

## インストール

```bash
# 最小構成（trigram検索のみ）
uv tool install jaqmd

# 形態素解析を使う場合
uv tool install "jaqmd[morph]"

# ベクトル検索を使う場合
uv tool install "jaqmd[vector]"

# Query Expansion を使う場合
uv tool install "jaqmd[qe]"

# MCP サーバーを使う場合
uv tool install "jaqmd[mcp]"

# すべての機能
uv tool install "jaqmd[all]"
```

> 注: ベクトル検索・リランク・Query Expansion で使用するモデルは初回実行時に Hugging Face から自動ダウンロードされ、`~/.cache/jaqmd/models/` にキャッシュされます（Query Expansion モデルは GGUF 形式で約3.4GB）。

## クイックスタート

```bash
# 1. コレクションを追加
jaqmd collection add ~/Documents/notes --name notes

# 2. インデックスを構築（trigram FTS）
jaqmd update

# 3. これだけで trigram 検索が使える
jaqmd search "形態素解析"

# 4. 形態素解析検索を有効化（任意）
jaqmd morph
jaqmd mosearch "形態素解析"

# 5. ベクトル検索を有効化（任意）
jaqmd embed
jaqmd vsearch "テキストの意味を捉えた検索"

# 6. すべてを組み合わせた高品質検索
jaqmd query "日本語の検索エンジンを作りたい"
```

## コマンドリファレンス

### インデックス構築

| コマンド | 処理内容 | 事前要件 |
|----------|----------|----------|
| `jaqmd collection add <path> --name <name>` | コレクションを追加 | - |
| `jaqmd update` | ファイルをスキャンし trigram FTS を構築 | - |
| `jaqmd morph` | 形態素解析して形態素 FTS を構築 | SudachiPy |
| `jaqmd embed [-f]` | ruri-v3 でチャンクをベクトル化 | fastembed |

### 検索

| コマンド | 検索方式 | 事前要件 |
|----------|----------|----------|
| `jaqmd search <query>` | trigram BM25 | `update` 済み |
| `jaqmd mosearch <query>` | 形態素 BM25 | `morph` 済み |
| `jaqmd vsearch <query>` | ベクトル意味検索 | `embed` 済み |
| `jaqmd query <query>` | ハイブリッド + リランク | `update` 済み（他は任意） |

### ドキュメント取得・管理

| コマンド | 処理内容 |
|----------|----------|
| `jaqmd get <file>[:line]` | パスまたは docid で1件取得 |
| `jaqmd multi-get <pattern>` | glob またはカンマ区切りで複数取得 |
| `jaqmd collection list` | コレクション一覧 |
| `jaqmd collection remove <name>` | コレクション削除 |
| `jaqmd ls [collection]` | コレクション内のファイル一覧 |
| `jaqmd status` | インデックスの構築状態を表示 |
| `jaqmd cleanup` | キャッシュ削除・DB最適化 |
| `jaqmd mcp` | MCP サーバーを stdio で起動（`--http` は未対応） |

### 検索オプション

```
-n <num>          結果件数（デフォルト: 5）
-c, --collection  コレクションを指定して絞り込み
--all             全件返却（--min-score と併用）
--min-score <num> スコア閾値
--full            全文を表示
--json            JSON 形式で出力
--md              Markdown 形式で出力
--xml             XML 形式で出力
--files           docid,score,filepath,context 形式で出力
--no-rerank       （query のみ）リランク を無効化し RRF 順のまま返す
--no-qe           （query のみ）Query Expansion を無効化し raw クエリのまま検索する
```

## 設定（環境変数・設定ファイル）

コマンドの挙動を変える設定値は、CLI 引数だけでなく環境変数・設定ファイルでも指定できます。優先順位は次のとおりです。

```
CLI 引数 > 環境変数 > 設定ファイル > 既定値
```

設定ファイルは `$XDG_CONFIG_HOME/jaqmd/config.toml`（`XDG_CONFIG_HOME` 未設定時は `~/.config/jaqmd/config.toml`）を TOML 形式で読み込みます。

```toml
[search]
n = 10
format = "md"       # plain|json|md|xml|files
full = false
min_score = 0.2
reranker = "default" # default|int8
rerank = true
qe = true

[index]
glob = "**/*.md"
batch_size = 4

[general]
quiet = false

[paths]
db = "/data/jaqmd/index.sqlite"
models = "/data/jaqmd/models"

[models]
embed = "sirasagi62/ruri-v3-310m-ONNX"
reranker = "szdr/ruri-v3-reranker-310m-onnx"
qe_repo = "szdr/jaqmd-qe-gemma-4-e2b-it"

[tuning]
rrf_k = 60
rerank_top_k = 50
```

### 検索コマンドの既定値（CLI フラグと対応）

| 環境変数 | 設定ファイル | 既定値 | 対応する CLI フラグ |
|---|---|---|---|
| `JAQMD_SEARCH_N` | `[search] n` | `5` | `-n` |
| `JAQMD_SEARCH_FORMAT` | `[search] format` | `plain` | `--json`/`--md`/`--xml`/`--files` |
| `JAQMD_SEARCH_FULL` | `[search] full` | `false` | `--full`/`--no-full` |
| `JAQMD_SEARCH_MIN_SCORE` | `[search] min_score` | (なし) | `--min-score` |
| `JAQMD_SEARCH_RERANKER` | `[search] reranker` | `default` | `--reranker`（query のみ） |
| `JAQMD_SEARCH_RERANK` | `[search] rerank` | `true` | `--no-rerank`（query のみ、無効化専用） |
| `JAQMD_SEARCH_QE` | `[search] qe` | `true` | `--no-qe`（query のみ、無効化専用） |
| `JAQMD_INDEX_GLOB` | `[index] glob` | `**/*.md` | `--glob`（collection add） |
| `JAQMD_INDEX_BATCH_SIZE` | `[index] batch_size` | `1` | `--batch-size`（embed） |
| `JAQMD_QUIET` | `[general] quiet` | `false` | `--quiet`/`--no-quiet`, `-q` |

`--no-rerank` / `--no-qe` は無効化専用のフラグです。設定ファイルや環境変数で `rerank`/`qe` を `false` にした場合、CLI から強制的に再度有効化するには `JAQMD_SEARCH_RERANK=true` / `JAQMD_SEARCH_QE=true` を都度指定してください（env は設定ファイルより優先されます）。

### エンジン設定（CLI フラグなし・env/設定ファイル専用）

モデル名・パス・チューニング値は CLI フラグを持たず、環境変数または設定ファイルでのみ変更できます。

| 環境変数 | 設定ファイル | 既定値 |
|---|---|---|
| `JAQMD_DB_PATH` | `[paths] db` | `$XDG_CACHE_HOME/jaqmd/index.sqlite` |
| `JAQMD_MODELS_DIR` | `[paths] models` | `$XDG_CACHE_HOME/jaqmd/models` |
| `JAQMD_MODELS_EMBED` | `[models] embed` | `sirasagi62/ruri-v3-310m-ONNX` |
| `JAQMD_MODELS_RERANKER` | `[models] reranker` | `szdr/ruri-v3-reranker-310m-onnx` |
| `JAQMD_MODELS_QE_REPO` | `[models] qe_repo` | `szdr/jaqmd-qe-gemma-4-e2b-it` |
| `JAQMD_TUNING_RRF_K` | `[tuning] rrf_k` | `60` |
| `JAQMD_TUNING_RERANK_TOP_K` | `[tuning] rerank_top_k` | `50` |

これらの値はプロセス起動時（`jaqmd` コマンド実行時）に一度だけ解決されます。

## 検索方式の使い分け

- **search**: 固有名詞・型番・コードなど正確なキーワードがわかっているとき。辞書不要で最速。
- **mosearch**: 表記ゆれを吸収したいとき、文法的なノイズを減らしたいとき。SudachiPy による正規化が効く。
- **vsearch**: 言い換えや概念的な検索。キーワードが一致しなくても意味で引ける。
- **query**: 品質を最優先するとき。利用可能なインデックスを自動判定して組み合わせ、ruri-reranker でリランクする。

## `jaqmd query` の動作

`query` は構築済みのインデックスを自動判定し、利用可能なものだけを組み合わせます。

```
query "クエリ"
  ├─ Query Expansion（lex: / vec: / hyde: に展開）
  ├─ search      （常に実行）
  ├─ mosearch    （morph 済みなら追加）
  ├─ vsearch     （embed 済みなら追加）
  ├─ RRF で融合
  └─ ruri-reranker でリランク
```

morph や embed が未実行でも `query` は動作します（その分、検索品質は段階的に向上します）。
`jaqmd[qe]` 未導入時も Query Expansion なし（raw クエリのみ）で degrade して動作します。

## 状態確認

```bash
$ jaqmd status
Collections : 2 (docs, notes)
Documents   : 1,234
─────────────────────────────
trigram FTS : ✓ 1,234 docs    (jaqmd update)
morph  FTS  : ✓ 1,234 docs    (jaqmd morph)
vectors     : ✗ not indexed   → run: jaqmd embed
─────────────────────────────
Available   : search, mosearch
Unavailable : vsearch, query(full)
```

## MCP サーバー

以下のツールセットを stdio トランスポートで公開します（`jaqmd[mcp]` が必要）。

| ツール | 内容 |
|--------|------|
| `query` | typed searches（`{type: lex\|vec\|hyde, text}` の配列、1〜10件）による RRF 融合 + リランク検索。先頭の search は融合時に2倍の重みを持つ |
| `get` | パスまたは docid（`abc123`。先頭に `#` は付けない）でドキュメントを1件取得 |
| `multi_get` | glob パターンまたはカンマ区切りで複数ドキュメントを取得 |
| `status` | インデックスの構築状態・コレクション一覧を取得 |

`query` は `jaqmd query` の単一クエリ文字列＋自動 Query Expansion とは異なり、MCP クライアント側が
lex/vec/hyde のサブクエリを明示的に組み立てて渡す設計です（tobi/qmd 準拠）。`vec`/`hyde` はベクトルインデックス
未構築の場合、その search のみ無視して degrade します。

起動:

```bash
jaqmd mcp
```

Claude Desktop（`~/Library/Application Support/Claude/claude_desktop_config.json`）:

```json
{
  "mcpServers": {
    "jaqmd": {
      "command": "jaqmd",
      "args": ["mcp"]
    }
  }
}
```

Claude Code（`.mcp.json` または `claude mcp add`）も同様に `command: jaqmd`, `args: ["mcp"]` で設定します。

> `--http` トランスポートは現時点で未対応です（`jaqmd mcp --http` はエラーで終了します）。

## トラブルシューティング

### reranker 実行時に `ValueError: setting an array element with a sequence` が出る

`jaqmd query` の reranker 段（`src/jaqmd/rerank.py`）で以下のような例外が出る場合、
reranker モデルのキャッシュが古い（修正前の `tokenizer.json` を含む）可能性があります。

```
ValueError: setting an array element with a sequence.
The requested array has an inhomogeneous shape after 1 dimensions.
```

キャッシュを削除して最新のモデルを再取得してください（次回 `jaqmd query` 実行時に自動DLされます）。

```bash
rm -rf ~/.cache/jaqmd/models/models--szdr--ruri-v3-reranker-310m-onnx
```

## 設計上の選択

- **言語**: Python（形態素解析・ベクトル推論のエコシステムが最も成熟しているため）
- **ストレージ**: SQLite 単一ファイル。FTS5（全文検索）と sqlite-vec（ベクトル）を同一DB内に保持
- **推論**: fastembed（ONNX Runtime）。PyTorch を引き込まず軽量
- **形態素解析**: SudachiPy。表記ゆれの正規化に強い
- **trigram**: SQLite FTS5 の trigram tokenizer を利用し、外部依存なしで実現

詳細な実装方針は [AGENTS.md](./AGENTS.md) を参照してください。


## Credits

- [Ruri](https://huggingface.co/cl-nagoya)
- Inspired by [tobi/qmd](https://github.com/tobi/qmd)
