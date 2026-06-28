from __future__ import annotations

import re


def make_snippet(body: str, terms: list[str], *, width: int = 80) -> str:
    """原文 body からマッチ箇所中心のスニペットを生成する。

    - terms のいずれかが最初に出現するオフセットを中心に前後 width//2 文字を切り出す。
    - どの term も見つからなければ先頭 width 文字にフォールバック。
    - 窓内の改行・連続空白は半角スペースに正規化し、1行で返す。
    """
    if not body:
        return ""

    # マッチ位置を探索（最小オフセット採用）
    best = len(body)
    lower_body = body.lower()
    for term in terms:
        if not term:
            continue
        idx = lower_body.find(term.lower())
        if idx != -1 and idx < best:
            best = idx

    if best == len(body):
        # フォールバック: 先頭
        center = 0
    else:
        center = best

    half = width // 2
    start = max(0, center - half)
    end = min(len(body), start + width)
    # 窓が後ろに余っていれば start を調整
    if end - start < width:
        start = max(0, end - width)

    chunk = body[start:end]
    # 改行・連続空白を正規化
    chunk = re.sub(r"\s+", " ", chunk).strip()

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(body) else ""
    return f"{prefix}{chunk}{suffix}"
