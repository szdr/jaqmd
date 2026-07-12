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
