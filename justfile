# 一覧
default:
    @just --list

# lint（チェックのみ）
lint:
    uv run ruff check src tests

# format チェック（CI 相当・書き換えなし）
fmt-check:
    uv run ruff format --check src tests

# 自動整形 + lint 自動修正
fix:
    uv run ruff check --fix src tests
    uv run ruff format src tests

# テスト（pyproject の addopts で integration は除外済み）
test:
    uv run pytest

# integration 含む全テスト
test-all:
    uv run pytest -m ""

# lint + fmt-check + test 一括（CI 相当）
check: lint fmt-check test

# リリース準備: 検証 → バージョン更新 → ビルド確認 → CHANGELOG 生成 → commit + タグ作成（push はしない）
# bump は patch / minor / major を指定（既定 patch）
release bump="patch":
    #!/usr/bin/env bash
    set -euo pipefail
    # 1. main ブランチ・作業ツリークリーン確認
    test "$(git rev-parse --abbrev-ref HEAD)" = "main" || { echo "main ブランチで実行してください"; exit 1; }
    git diff --quiet && git diff --cached --quiet || { echo "未コミットの変更があります"; exit 1; }
    # 2. origin/main と同期確認（main が最新であること）
    git fetch origin
    test "$(git rev-parse @)" = "$(git rev-parse @{u})" || { echo "main が origin/main と同期していません"; exit 1; }
    # 3. リリース前検証（lint + fmt-check + test）
    just check
    # 4. バージョン更新
    uv version --bump {{bump}}
    VERSION="$(uv version --short)"
    echo "バージョン: v${VERSION}"
    # 5. ローカルビルド確認
    uv build
    # 6. CHANGELOG 生成（未タグのコミットを新バージョンとして反映）
    uvx git-cliff --tag "v${VERSION}" -o CHANGELOG.md
    # 7. commit + 注釈付きタグ作成（push はしない）
    git add pyproject.toml uv.lock CHANGELOG.md
    git commit -m "chore(release): v${VERSION}"
    git tag -a "v${VERSION}" -m "v${VERSION}"
    echo "準備完了。内容を確認し 'just release-push' で公開してください"

# リリース公開: main とタグを push → GitHub Actions が PyPI 公開・GitHub Release 作成
release-push:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION="$(uv version --short)"
    git push origin main
    git push origin "v${VERSION}"
