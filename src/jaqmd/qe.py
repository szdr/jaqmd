from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .progress import NULL_REPORTER, ProgressReporter
from .store import get_qe_cache, set_qe_cache

QE_MODEL_REPO = "szdr/jaqmd-qe-gemma-4-e2b-it"
QE_MODEL_FILE = "gguf/gemma-4-e2b-it.Q4_K_M.gguf"
QE_MODEL_ID = QE_MODEL_REPO  # qe_cache.model_id に記録する識別子

_llm = None
_llm_load_attempted = False


@contextmanager
def _suppress_native_stderr():
    """llama.cpp のネイティブコードが fd レベルで stderr に吐くログを抑制する。"""
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # 実 fd を持たない（テスト等でキャプチャされた）場合は何もしない
        yield
        return
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(devnull)
        os.close(saved)


@dataclass
class ExpansionResult:
    lex: list[str]
    vec: str
    hyde: str


def _get_llm():
    """Query Expansion 用の LLM（llama.cpp）をロードして返す。

    fastembed 未導入時（rerank.py の _get_encoder）と同じ「1度だけ試行し
    失敗したら以後 None を返す」パターン。GGUF は HF hub から自動DLされる。
    """
    global _llm, _llm_load_attempted
    if _llm is not None:
        return _llm
    if _llm_load_attempted:
        return None
    _llm_load_attempted = True

    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "警告: llama-cpp-python が見つかりません。Query Expansion を無効化して続行します。\n"
            "→ pip install 'jaqmd[qe]' を実行すると Query Expansion が有効になります。",
            file=sys.stderr,
        )
        return None

    try:
        cache_dir = Path.home() / ".cache" / "jaqmd" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings(), _suppress_native_stderr():
            warnings.simplefilter("ignore")
            _llm = Llama.from_pretrained(
                repo_id=QE_MODEL_REPO,
                filename=QE_MODEL_FILE,
                cache_dir=str(cache_dir),
                n_ctx=2048,
                verbose=False,
            )
    except Exception as e:
        print(
            f"警告: Query Expansion モデルのロードに失敗しました（{e}）。無効化して続行します。",
            file=sys.stderr,
        )
        return None
    return _llm


def _extract_json(text: str) -> Optional[dict]:
    """応答テキストから最初の '{' に対応する '}' までを抜き出して JSON デコードする。

    モデルは thinking なしで学習済みだが、前後に説明文やコードフェンスが
    混入しても壊れないよう、波括弧の対応を数えて JSON 部分のみを抽出する。
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _run_model(llm, query_text: str) -> Optional[ExpansionResult]:
    """LLM を実行し、応答を ExpansionResult にパースする。失敗時は None。"""
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": query_text}],
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(
            f"警告: Query Expansion の推論に失敗しました（{e}）。無効化して続行します。",
            file=sys.stderr,
        )
        return None

    obj = _extract_json(content)
    if obj is None:
        return None

    lex = obj.get("lex")
    vec = obj.get("vec")
    hyde = obj.get("hyde")
    if not isinstance(lex, list) or not isinstance(vec, str) or not isinstance(hyde, str):
        return None

    return ExpansionResult(lex=[str(t) for t in lex], vec=vec, hyde=hyde)


def expand(
    conn: sqlite3.Connection,
    query_text: str,
    *,
    reporter: Optional[ProgressReporter] = None,
) -> Optional[ExpansionResult]:
    """クエリを lex/vec/hyde に展開する。qe_cache を優先参照する。

    llama-cpp-python 未導入・モデルロード失敗・推論失敗・応答パース失敗の
    いずれの場合も None を返し、呼び出し側は raw クエリへ degrade する。
    """
    reporter = reporter or NULL_REPORTER
    query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()

    cached = get_qe_cache(conn, query_hash, QE_MODEL_ID)
    if cached is not None:
        try:
            lex = json.loads(cached["lex_query"]) if cached["lex_query"] else []
        except json.JSONDecodeError:
            lex = []
        return ExpansionResult(
            lex=lex,
            vec=cached["vec_query"] or "",
            hyde=cached["hyde_text"] or "",
        )

    llm = _get_llm()
    if llm is None:
        return None

    with reporter.step("Query Expansion"):
        result = _run_model(llm, query_text)

    if result is None:
        return None

    set_qe_cache(
        conn,
        query_hash,
        query_text,
        json.dumps(result.lex, ensure_ascii=False),
        result.vec,
        result.hyde,
        QE_MODEL_ID,
    )
    conn.commit()

    return result
