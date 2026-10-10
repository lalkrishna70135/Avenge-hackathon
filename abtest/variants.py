"""Turn a one-line variant diff into a rewritten section of the base prompt, using Sarvam.

The VANI base prompt is ~160 KB, so only the section the diff talks about is sent: the window of
lines that shares the most words with the diff. The model returns that section rewritten, and the
result is saved as a file so experiment configs (and their hashes) never depend on a live API call.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "sarvam-105b"
WINDOW_LINES = 30

SYSTEM = (
    "You edit system prompts for VANI, IndiaMART's buyer-side voice bot. You are given one section of "
    "the current prompt and a requested change. Rewrite the section so it implements the change and "
    "nothing else. Where an existing rule in the section conflicts with the change, edit that rule so the "
    "section is consistent; do not just paste the change in as a new sentence. Keep every other rule, the tone, and any template syntax such as {%if ...%} exactly "
    "as it is. Return the COMPLETE section, from its first line to its last, including every line you did not "
    "need to change. Reply with the rewritten section text only: no preamble, no explanation, no markdown fences."
)

_STOP = set("a an and the to of in on for with or is are be it as at by any each once this that then their "
            "before after into from use used only never always".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2 and w not in _STOP}


def find_section(base_prompt: str, diff: str, window: int = WINDOW_LINES, anchor: str | None = None) -> tuple[int, int]:
    """[start, end) line range of the base prompt to rewrite. With `anchor`, the window starts at
    the first line containing it. Otherwise it is the window that best matches the diff, with
    words weighted by rarity across the prompt's lines."""
    lines = base_prompt.splitlines()
    if anchor:
        for i, line in enumerate(lines):
            if anchor in line:
                return i, min(i + window, len(lines))
        raise ValueError(f"anchor not found in base prompt: {anchor!r}")
    line_words = [_words(l) for l in lines]
    df = Counter(w for ws in line_words for w in ws)
    target = _words(diff)
    idf = {w: math.log((len(lines) + 1) / (df[w] + 1)) for w in target}
    best, best_start = -1.0, 0
    for start in range(0, max(len(lines) - window, 0) + 1):
        seen = set().union(*line_words[start:start + window]) & target
        score = sum(idf[w] for w in seen)
        if score > best:
            best, best_start = score, start
    return best_start, min(best_start + window, len(lines))


def load_env_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not line.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SystemExit("SARVAM_API_KEY is not set. Put it in .env.")
    return key


def generate(base_prompt: str, diff: str, anchor: str | None = None, window: int | None = None) -> dict:
    from sarvamai import SarvamAI

    start, end = find_section(base_prompt, diff, window or WINDOW_LINES, anchor)
    section = "\n".join(base_prompt.splitlines()[start:end])
    client = SarvamAI(api_subscription_key=load_env_key())
    resp = client.chat.completions(
        model=MODEL,
        temperature=0.2,
        reasoning_effort="low",
        max_tokens=16000,             # sarvam-105b reasons before answering; that counts against this
        request_options={"timeout_in_seconds": 300, "max_retries": 2},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CURRENT SECTION:\n{section}\n\nREQUESTED CHANGE:\n{diff}"},
        ],
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length":
        raise RuntimeError(f"Sarvam reply was cut off at max_tokens (usage={resp.usage})")
    if not text:
        raise RuntimeError(f"Sarvam returned no text (finish_reason={choice.finish_reason}, usage={resp.usage})")
    text = re.sub(r"^```\w*\n|\n```$", "", text)   # in case the model fences it anyway
    return {"model": MODEL, "lines": [start + 1, end], "original": section, "rewritten": text}
