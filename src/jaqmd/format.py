from __future__ import annotations

import json
import re
import shutil
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from .search.trisearch import SearchResult


def format_results(
    results: list["SearchResult"],
    *,
    fmt: str = "default",
    full: bool = False,
    query: str = "",
    color: bool = False,
) -> str:
    if not results:
        return "No results found."

    dispatch = {
        "json": lambda r: _fmt_json(r, full=full),
        "md": lambda r: _fmt_md(r, full=full),
        "xml": _fmt_xml,
        "files": _fmt_files,
    }
    fn = dispatch.get(fmt)
    if fn:
        return fn(results)
    return _fmt_default(results, full=full, query=query, color=color)


def _fmt_default(
    results: list, *, full: bool, query: str = "", color: bool = False
) -> str:
    if not color:
        # 非 TTY（パイプ・リダイレクト・テスト）では無着色の従来出力を維持する
        lines = []
        for r in results:
            label = r.title or r.filepath
            lines.append(f"[{r.docid}] {label}  (score: {r.score:.3f})")
            lines.append(f"  {r.filepath}")
            text = r.body if full else r.snippet
            lines.append(f"  {text}")
            lines.append("")
        return "\n".join(lines).rstrip()

    terms = _query_terms(query)
    rule = typer.style("─" * _rule_width(), dim=True)
    blocks = []
    for r in results:
        label = r.title or r.filepath
        header = (
            typer.style(f"[{r.docid}]", fg="cyan", bold=True)
            + " "
            + typer.style(label, bold=True)
            + "  "
            + typer.style(f"(score: {r.score:.3f})", dim=True)
        )
        path = "  " + typer.style(r.filepath, fg="blue")
        text = r.body if full else r.snippet
        snippet = "  " + _highlight(text, terms, color=True)
        blocks.append("\n".join([header, path, snippet]))
    return f"\n{rule}\n".join(blocks)


def _rule_width() -> int:
    cols = shutil.get_terminal_size((80, 20)).columns
    return min(cols, 60)


def _query_terms(query: str) -> list[str]:
    # 空白区切りで語を取り出す（日本語のスペースなしクエリは全体が1語になる）
    return [t for t in query.split() if len(t) >= 2]


def _highlight(text: str, terms: list[str], *, color: bool) -> str:
    if not color or not terms:
        return text
    # 長い語を優先して1パスで置換し、入れ子・二重着色を避ける
    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)),
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: typer.style(m.group(0), fg="yellow", bold=True), text)


def _fmt_json(results: list, *, full: bool = False) -> str:
    data = []
    for r in results:
        entry: dict = {
            "docid": r.docid,
            "score": r.score,
            "filepath": r.filepath,
            "title": r.title,
            "snippet": r.snippet,
        }
        if full:
            entry["body"] = r.body
        data.append(entry)
    return json.dumps(data, ensure_ascii=False, indent=2)


def _fmt_md(results: list, *, full: bool) -> str:
    lines = []
    for r in results:
        lines.append(f"## {r.title or r.filepath}")
        lines.append(f"- **path**: `{r.filepath}`")
        lines.append(f"- **docid**: `{r.docid}`")
        lines.append(f"- **score**: {r.score:.3f}")
        lines.append("")
        text = r.body if full else r.snippet
        lines.append(text)
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
