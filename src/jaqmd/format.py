from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .search.trisearch import SearchResult


def format_results(
    results: list["SearchResult"],
    *,
    fmt: str = "default",
    full: bool = False,
) -> str:
    if not results:
        return "No results found."

    dispatch = {
        "json": _fmt_json,
        "md": lambda r: _fmt_md(r, full=full),
        "xml": _fmt_xml,
        "files": _fmt_files,
    }
    fn = dispatch.get(fmt)
    if fn:
        return fn(results)
    return _fmt_default(results, full=full)


def _fmt_default(results: list, *, full: bool) -> str:
    lines = []
    for r in results:
        label = r.title or r.filepath
        lines.append(f"[{r.docid}] {label}  (score: {r.score:.3f})")
        lines.append(f"  {r.filepath}")
        lines.append(f"  {r.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _fmt_json(results: list) -> str:
    data = [
        {
            "docid": r.docid,
            "score": r.score,
            "filepath": r.filepath,
            "title": r.title,
            "snippet": r.snippet,
        }
        for r in results
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def _fmt_md(results: list, *, full: bool) -> str:
    lines = []
    for r in results:
        lines.append(f"## {r.title or r.filepath}")
        lines.append(f"- **path**: `{r.filepath}`")
        lines.append(f"- **docid**: `{r.docid}`")
        lines.append(f"- **score**: {r.score:.3f}")
        lines.append("")
        lines.append(r.snippet)
        lines.append("")
    return "\n".join(lines).rstrip()


def _fmt_xml(results: list) -> str:
    lines = ["<results>"]
    for r in results:
        lines.append("  <result>")
        lines.append(f"    <docid>{_xe(r.docid)}</docid>")
        lines.append(f"    <score>{r.score:.3f}</score>")
        lines.append(f"    <filepath>{_xe(r.filepath)}</filepath>")
        lines.append(f"    <title>{_xe(r.title)}</title>")
        lines.append(f"    <snippet>{_xe(r.snippet)}</snippet>")
        lines.append("  </result>")
    lines.append("</results>")
    return "\n".join(lines)


def _fmt_files(results: list) -> str:
    lines = []
    for r in results:
        lines.append(f"{r.docid},{r.score:.3f},{r.filepath},{r.snippet}")
    return "\n".join(lines)


def _xe(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
