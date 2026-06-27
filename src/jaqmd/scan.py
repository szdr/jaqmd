from __future__ import annotations

import hashlib
from pathlib import Path

_SUPPORTED = {".md", ".txt"}


def extract_title(body: str, rel_path: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return Path(rel_path).stem


def scan_collection(
    collection_path: str,
    glob_mask: str = "**/*.md",
) -> list[dict]:
    """コレクションディレクトリをスキャンしてファイル情報のリストを返す。"""
    base = Path(collection_path)
    results = []

    for p in base.glob(glob_mask):
        if not p.is_file() or p.suffix.lower() not in _SUPPORTED:
            continue
        try:
            body = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        rel = str(p.relative_to(base))
        results.append(
            {
                "path": rel,
                "body": body,
                "title": extract_title(body, rel),
                "mtime": int(p.stat().st_mtime),
                "hash": hashlib.sha256(body.encode()).hexdigest(),
            }
        )

    return results
