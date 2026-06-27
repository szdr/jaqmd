# jaqmd

日本語ドキュメントに特化したローカル検索エンジン。

[tobi/qmd](https://github.com/tobi/qmd) からインスピレーションを受けつつ、日本語テキストの検索品質に振り切った設計を採用しています。

## 特徴

- **日本語特化**: bigram検索・形態素解析検索・ベクトル検索の3つを日本語向けに最適化
- **完全ローカル**: SQLite + sqlite-vec + FTS5 で完結。外部DBやサーバー不要
- **薄い依存**: PyTorch非依存。ONNX Runtime ベースの軽量構成
- **段階的なインデックス構築**: 必要な検索機能だけを段階的に有効化できる

## アーキテクチャ概要

```
ドキュメント ──► jaqmd update ──► bigram FTS（常時）
                 jaqmd morph  ──► 形態素 FTS（オプション）
                 jaqmd embed  ──► ベクトル（オプション）

検索:
  jaqmd bisearch  bigram BM25検索
  jaqmd mosearch  形態素解析 BM25検索
  jaqmd vsearch   ベクトル意味検索（ruri-v3）
  jaqmd query     上記を組み合わせ + リランク（推奨）
```

## モデル構成

| 役割 | モデル | 提供元 | ライセンス |
|------|--------|--------|-----------|
| Embedding | `cl-nagoya/ruri-v3-310m` | 名古屋大学 | Apache-2.0 |
| Reranker | `cl-nagoya/ruri-v3-reranker-310m` | 名古屋大学 | Apache-2.0 |
| Query Expansion | jaqmd 独自開発（予定） | jaqmd | - |

いずれも ONNX 形式で fastembed 経由で読み込みます。

## インストール

```bash
# 最小構成（bigram検索のみ）
pip install jaqmd

# 形態素解析を使う場合
pip install "jaqmd[morph]"

# ベクトル検索を使う場合
pip install "jaqmd[vector]"

# すべての機能
pip install "jaqmd[all]"
```

> 注: ベクトル検索・リランクで使用する ONNX モデルは初回実行時に Hugging Face から自動ダウンロードされ、`~/.cache/jaqmd/models/` にキャッシュされます。

## クイックスタート

```bash
# 1. コレクションを追加
jaqmd collection add ~/Documents/notes --name notes

# 2. インデックスを構築（bigram FTS）
jaqmd update

# 3. これだけで bigram 検索が使える
jaqmd bisearch "形態素解析"

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
| `jaqmd update [--pull]` | ファイルをスキャンし bigram FTS を構築 | - |
| `jaqmd morph` | 形態素解析して形態素 FTS を構築 | SudachiPy |
| `jaqmd embed [-f]` | ruri-v3 でチャンクをベクトル化 | fastembed |

### 検索

| コマンド | 検索方式 | 事前要件 |
|----------|----------|----------|
| `jaqmd bisearch <query>` | bigram BM25 | `update` 済み |
| `jaqmd mosearch <query>` | 形態素 BM25 | `morph` 済み |
| `jaqmd vsearch <query>` | ベクトル意味検索 | `embed` 済み |
| `jaqmd query <query>` | ハイブリッド + リランク | `update` 済み（他は任意） |
| `jaqmd search <query>` | `bisearch` のエイリアス（qmd互換） | `update` 済み |

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
| `jaqmd mcp [--http]` | MCP サーバーを起動 |

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
```

## 検索方式の使い分け

- **bisearch**: 固有名詞・型番・コードなど正確なキーワードがわかっているとき。辞書不要で最速。
- **mosearch**: 表記ゆれを吸収したいとき、文法的なノイズを減らしたいとき。SudachiPy による正規化が効く。
- **vsearch**: 言い換えや概念的な検索。キーワードが一致しなくても意味で引ける。
- **query**: 品質を最優先するとき。利用可能なインデックスを自動判定して組み合わせ、ruri-reranker でリランクする。

## `jaqmd query` の動作

`query` は構築済みのインデックスを自動判定し、利用可能なものだけを組み合わせます。

```
query "クエリ"
  ├─ Query Expansion（lex: / vec: / hyde: に展開）
  ├─ bisearch    （常に実行）
  ├─ mosearch    （morph 済みなら追加）
  ├─ vsearch     （embed 済みなら追加）
  ├─ RRF で融合
  └─ ruri-reranker でリランク
```

morph や embed が未実行でも `query` は動作します（その分、検索品質は段階的に向上します）。

## 状態確認

```bash
$ jaqmd status
Collections : 2 (docs, notes)
Documents   : 1,234
─────────────────────────────
bigram FTS  : ✓ 1,234 docs    (jaqmd update)
morph  FTS  : ✓ 1,234 docs    (jaqmd morph)
vectors     : ✗ not indexed   → run: jaqmd embed
─────────────────────────────
Available   : bisearch, mosearch
Unavailable : vsearch, query(full)
```

## 設計上の選択

- **言語**: Python（形態素解析・ベクトル推論のエコシステムが最も成熟しているため）
- **ストレージ**: SQLite 単一ファイル。FTS5（全文検索）と sqlite-vec（ベクトル）を同一DB内に保持
- **推論**: fastembed（ONNX Runtime）。PyTorch を引き込まず軽量
- **形態素解析**: SudachiPy。表記ゆれの正規化に強い
- **bigram**: SQLite FTS5 の trigram tokenizer を利用し、外部依存なしで実現

詳細な実装方針は [AGENTS.md](./AGENTS.md) を参照してください。


## Credits

- [Ruri](https://huggingface.co/cl-nagoya)
- Inspired by [tobi/qmd](https://github.com/tobi/qmd)