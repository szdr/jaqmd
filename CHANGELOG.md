## [0.1.8] - 2026-07-24

### 🚀 Features

- MCP の query → get/multi_get の受け渡し不整合を解消
## [0.1.7] - 2026-07-22

### 🚀 Features

- リランクのピークメモリを抑制する設定を追加
## [0.1.6] - 2026-07-20

### 🚀 Features

- Query の統合検索スコアを tobi/qmd 式の位置依存ブレンドに寄せる
## [0.1.5] - 2026-07-20

### 🚀 Features

- 検索結果 snippet を長く・設定可能にする
## [0.1.4] - 2026-07-15

### 🚜 Refactor

- 未使用の --pull オプションと不要な記述を削除
## [0.1.3] - 2026-07-14

### 🚀 Features

- --version オプションを追加
- --help を -h でも呼べるようにする

### ⚙️ Miscellaneous Tasks

- PLANS.md を削除
## [0.1.2] - 2026-07-14

### ⚙️ Miscellaneous Tasks

- Pyproject に readme を指定
## [0.1.1] - 2026-07-13

### 🚀 Features

- Implement morph and mosearch functionalities with tokenization support
- Enhance search results with original text and snippet generation
- Add vector search functionality and snippet extraction improvements
- Implement hybrid search functionality with RRF fusion and query enhancements
- Add min-max scaling for RRF scores in query results
- Implement reranker functionality with optional disabling
- Add progress reporting functionality
- Add collection filtering to update and morph commands, and implement get_collection function
- Add progress reporting and quiet option to update, morph, and embed commands
- Update model references and enhance embedding functionality
- Add --force option to morph command for full index rebuild
- Add query expansion feature with llama-cpp-python integration
- Add query expansion result formatting and callback support
- Suppress stderr output during LLM loading
- Add reranker model option and update rerank functionality
- Visualize model loading progress during first-run query
- MCP サーバー対応（tobi/qmd 準拠）
- Env / 設定ファイルによるコマンド既定値の上書きに対応
- Use ruff, just
- GitHub Actions による CI ワークフローを追加
- リリースフローを構築

### 🐛 Bug Fixes

- Ensure idempotent schema application and enhance database initialization
- 長文入力時のSudachi分割エラーを回避
- Setup-uv のバージョン指定を存在する v7 タグに修正
- MCP get の docid 指定で "#" 誤付与を解消

### 💼 Other

- Trigram

### 📚 Documentation

- PLANS.md の完了表記を実装状況に合わせて更新

### ⚙️ Miscellaneous Tasks

- Format code and update dependencies
