"""Speak a conversation with Sarvam TTS (bulbul:v3): one voice for the bot, one for the buyer.

Each turn is synthesised separately at a fixed sample rate and the WAVs are joined with the stdlib
`wave` module, with a short silence between turns.
"""
from __future__ import annotations

import base64
import io
import re
import time
import wave
from pathlib import Path

from . import sarvam

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "runs" / "audio"
TTS_MODEL = "bulbul:v3"
RATE = 22050
BOT_VOICE = "priya"
BUYER_VOICES = {"male": "rahul", "female": "kavya", "unknown": "rahul"}
MAX_CHARS = 450                      # keep each request short; long turns are split on sentence ends
GAP_S = 0.35
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def language_for(text: str, hint: str | None = None) -> str:
    if _DEVANAGARI.search(text):
        return "hi-IN"
    if hint and hint.lower() in ("hindi", "hinglish", "hi-in"):
        return "hi-IN"
    return "en-IN"


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?।])\s+", text.strip()):
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
        while len(cur) > limit:                      # a single very long sentence
            cut = cur.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            parts.append(cur[:cut])
            cur = cur[cut:].strip()
    if cur:
        parts.append(cur)
    return parts


def synth(text: str, speaker: str, language: str, tries: int = 6) -> list[bytes]:
    """WAV bytes per chunk of `text`. Rate limits and server errors are retried with backoff, like chat."""
    out = []
    for chunk in split_text(text):
        delay = 2.0
        for attempt in range(tries):
            try:
                r = sarvam.client().text_to_speech.convert(text=chunk, language_code=language, speaker=speaker,
                                                           model=TTS_MODEL, speech_sample_rate=RATE)
                break
            except Exception as exc:      # noqa: BLE001  (SDK raises many types)
                if attempt + 1 < tries and sarvam._retryable(exc):
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise
        out += [base64.b64decode(a) for a in r.audios]
    return out


def join_wavs(clips: list[bytes], gap_s: float = GAP_S) -> bytes:
    """Concatenate WAV clips that share sample rate, width and channels; raise if they don't."""
    if not clips:
        raise ValueError("no audio clips")
    params, frames = None, []
    for clip in clips:
        with wave.open(io.BytesIO(clip)) as w:
            p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = p
            elif p != params:
                raise ValueError(f"clip format {p} differs from {params}")
            frames.append(w.readframes(w.getnframes()))
    ch, width, rate = params
    silence = b"\x00" * int(rate * gap_s) * width * ch
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(silence.join(frames))
    return buf.getvalue()


def render_conversation(turns: list[dict], buyer_gender: str = "unknown", language_hint: str | None = None) -> bytes:
    clips = []
    for t in turns:
        if not t["text"].strip():
            continue
        speaker = BOT_VOICE if t["speaker"] == "bot" else BUYER_VOICES.get(buyer_gender, "rahul")
        clips += synth(t["text"], speaker, language_for(t["text"], language_hint))
    return join_wavs(clips)


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()
