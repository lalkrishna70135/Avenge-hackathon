"""Shared Sarvam access for the prompt lab: chat with a disk cache, JSON parsing, a thread pool.

Every chat reply is cached under data/sarvam/cache/<sha256 of the request>.json, written
atomically, so a stage can be stopped and resumed without paying for finished work and the
dashboard can replay everything offline. Change any request field and the key changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from .variants import load_env_key

ROOT = Path(__file__).resolve().parent.parent
SARVAM_DIR = ROOT / "data" / "sarvam"
CACHE_DIR = SARVAM_DIR / "cache"
MODEL = "sarvam-105b"                        # reasons before answering: slower (~40 s), used for judging
FAST_MODEL = "sarvam-105b-conversations"     # no reasoning step (~1 s): used for bot / buyer turns and personas

class EmptyReply(RuntimeError):
    """The model returned no text even with a larger token budget."""


_client = None
_lock = threading.Lock()


def client():
    """Created on first use, so code paths that never call Sarvam need no key."""
    global _client
    with _lock:
        if _client is None:
            from sarvamai import SarvamAI
            _client = SarvamAI(api_subscription_key=load_env_key())
        return _client


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def cache_key(**request) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_json(text: str):
    """JSON from a model reply that may be fenced or wrapped in prose."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # only for a reply that failed to parse: the persona template's "<..., or null>" slots sometimes come
        # back as `null"`, so drop that stray quote (valid JSON such as "was null", is never touched)
        text = re.sub(r'\bnull"(\s*[,}\]])', r"null\1", text)
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    return code in (429, 500, 502, 503, 504) or type(exc).__name__ in ("ReadTimeout", "ConnectTimeout",
                                                                       "ConnectError", "RemoteProtocolError")


def chat(messages: list[dict], *, temperature: float = 0.3, max_tokens: int = 4000, reasoning_effort: str = "low",
         as_json: bool = False, model: str = MODEL, cache: bool = True, tries: int = 8, replicate: int = 0):
    """One chat completion. Returns the reply text (or parsed JSON with as_json=True).
    An empty or length-truncated reply is retried with double, then quadruple max_tokens:
    sarvam-105b's reasoning tokens count against max_tokens and occasionally use all of it.
    `replicate` > 0 only changes the cache key: an independent sample of an otherwise identical request."""
    request = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                   reasoning_effort=reasoning_effort)
    if replicate:
        request["replicate"] = replicate
    path = CACHE_DIR / f"{cache_key(**request)}.json"
    if cache and path.exists():
        text = json.loads(path.read_text())["text"]
        return parse_json(text) if as_json else text

    budget, delay = max_tokens, 3.0
    for attempt in range(tries):
        try:
            resp = client().chat.completions(
                model=model, messages=messages, temperature=temperature, max_tokens=budget,
                reasoning_effort=reasoning_effort, request_options={"timeout_in_seconds": 300, "max_retries": 0})
        except Exception as exc:          # noqa: BLE001  (SDK raises many types)
            if attempt + 1 < tries and _retryable(exc):
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if (choice.finish_reason == "length" or not text) and budget < 4 * max_tokens:
            budget *= 2
            continue
        if choice.finish_reason == "length" or not text:
            raise EmptyReply(f"Sarvam reply empty or cut off (finish_reason={choice.finish_reason})")
        result = parse_json(text) if as_json else text     # parse before caching: never cache unusable JSON
        if cache:
            usage = resp.usage
            write_atomic(path, json.dumps({"request": request, "text": text,
                                           "usage": {"prompt": getattr(usage, "prompt_tokens", None),
                                                     "completion": getattr(usage, "completion_tokens", None)}},
                                          ensure_ascii=False))
        return result
    raise RuntimeError("Sarvam chat failed after retries")


def run_pool(fn: Callable, items: Iterable, workers: int = 6, label: str = "items",
             every: int = 10) -> tuple[list, list]:
    """Run fn(item) in a thread pool. One failure never stops the stage: failures are returned
    as (item, error) and progress is printed every `every` completions."""
    items = list(items)
    done, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                done.append(fut.result())
            except Exception as exc:      # noqa: BLE001
                failed.append((futures[fut], f"{type(exc).__name__}: {str(exc)[:300]}"))
            if i % every == 0 or i == len(items):
                print(f"  {label}: {i}/{len(items)} done, {len(failed)} failed", flush=True)
    return done, failed


def usage_totals() -> dict:
    """Token totals across the cache, for cost reporting."""
    p = c = n = 0
    for f in CACHE_DIR.glob("*.json"):
        u = json.loads(f.read_text()).get("usage") or {}
        p += u.get("prompt") or 0
        c += u.get("completion") or 0
        n += 1
    return {"calls": n, "prompt_tokens": p, "completion_tokens": c}
