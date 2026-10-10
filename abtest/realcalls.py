"""Real VANI recordings -> diarized transcripts (Sarvam batch STT) -> quality-matrix QA + buyer personas.

The recordings do not map to call-log rows, so they are used for call-quality analysis and to seed
realistic simulated buyers, not to replace the call log.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import sarvam

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "recordings"
TRANSCRIPT_DIR = sarvam.SARVAM_DIR / "transcripts"
REAL_CALLS = sarvam.SARVAM_DIR / "real_calls.jsonl"
MATRIX = (ROOT / "data" / "call_quality_matrix.md")
MIN_BYTES = 20_000          # shorter files are ring/hang-up fragments
STT_MODEL = "saaras:v3"

PARAMS = ["Outcome Accuracy", "Quantity Accuracy (Numeric)", "Quantity Unit Accuracy", "Quantity Probing Error",
          "Qty Missed Opportunity", "Spec1 Accuracy", "Spec2 Accuracy", "Spec Missed Opportunity",
          "Looping Behavior", "WER", "Additional_details", "OnCall AstBuy Error",
          "Product Change / No Product Accuracy", "Category ISQ's Error", "MCAT / Search errors", "Dead air"]
_BOT_MARKERS = re.compile(r"india\s*mart|इंडिया\s*मार्ट|indiamart|requirement|रिक्वायरमेंट", re.I)


# ---------------------------------------------------------------------------- transcription
def transcribe(paths: list[Path], out_dir: Path = TRANSCRIPT_DIR, chunk: int = 20) -> list[str]:
    """Batch STT with diarization. Skips files already transcribed. Returns error strings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (out_dir / f"{p.name}.json").exists()]
    errors = []
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        try:
            job = sarvam.client().speech_to_text_job.create_job(
                model=STT_MODEL, mode="codemix", with_diarization=True, num_speakers=2, language_code="unknown")
            job.upload_files(file_paths=[str(p) for p in part])
            job.start()
            job.wait_until_complete()
            job.download_outputs(output_dir=str(out_dir))
        except Exception as exc:          # noqa: BLE001
            errors.append(f"chunk {i // chunk}: {type(exc).__name__}: {str(exc)[:200]}")
        missing = [p.name for p in part if not (out_dir / f"{p.name}.json").exists()]
        errors += [f"{m}: no transcript" for m in missing]
        print(f"  STT: {min(i + chunk, len(todo))}/{len(todo)} files, {len(errors)} problems", flush=True)
    return errors


def turns(transcript: dict) -> list[dict]:
    """Diarized entries -> [{speaker: 'bot'|'buyer', text, start, end}], merging consecutive same-speaker entries."""
    entries = (transcript.get("diarized_transcript") or {}).get("entries") or []
    if not entries:
        return []
    bot = bot_speaker(entries)
    out = []
    for e in entries:
        who = "bot" if str(e.get("speaker_id")) == bot else "buyer"
        text = (e.get("transcript") or "").strip()
        if not text:
            continue
        if out and out[-1]["speaker"] == who:
            out[-1]["text"] += " " + text
            out[-1]["end"] = e.get("end_time_seconds", out[-1]["end"])
        else:
            out.append({"speaker": who, "text": text, "start": e.get("start_time_seconds", 0.0),
                        "end": e.get("end_time_seconds", 0.0)})
    return out


def bot_speaker(entries: list[dict]) -> str:
    """Speaker ids are arbitrary per call (0/1 on one, 1/2 on another). The bot is whoever mentions
    IndiaMART / 'requirement' most; ties go to whoever has the most words (the bot talks more)."""
    score, words = {}, {}
    for e in entries:
        s, t = str(e.get("speaker_id")), e.get("transcript") or ""
        score[s] = score.get(s, 0) + len(_BOT_MARKERS.findall(t))
        words[s] = words.get(s, 0) + len(t.split())
    return max(score, key=lambda s: (score[s], words[s]))


def dead_air(ts: list[dict], threshold: float = 5.0) -> dict:
    gaps = [b["start"] - a["end"] for a, b in zip(ts, ts[1:]) if b["start"] - a["end"] > threshold]
    return {"gaps_over_5s": len(gaps), "longest_gap_s": round(max(gaps), 1) if gaps else 0.0}


def buyer_said_something(ts: list[dict]) -> bool:
    words = " ".join(t["text"] for t in ts if t["speaker"] == "buyer").lower()
    words = re.sub(r"hello|hallo|हेलो|हैलो|हलो|[^\w\s]", " ", words).split()
    return len(words) >= 3


def render(ts: list[dict]) -> str:
    return "\n".join(f"{'BOT' if t['speaker'] == 'bot' else 'BUYER'}: {t['text']}" for t in ts)


# ---------------------------------------------------------------------------- judge (shared with simulated calls)
JUDGE_SYSTEM = (
    "You audit calls made by VANI, IndiaMART's buyer-side voice bot, using the VANI Call Quality Matrix below. "
    "You see only a transcript. Judge each parameter as Pass, Non-fatal, Fatal or N/A, with a short reason. "
    "Use N/A for anything a text transcript cannot show (WER, Dead air, MCAT / Search errors) and for parameters "
    "that do not apply to this call. Do not guess.\n\n" + (MATRIX.read_text() if MATRIX.exists() else "")
)
JUDGE_SCHEMA = (
    'Reply with JSON only:\n{"lead_created": true|false, "disposition": "BL Generated|BL Enriched|BL Approved|'
    'BL Deleted|No Product Requirement|Not Interested|Call Dropped|Other", "buyer_intent": "buying|not buying|unclear", '
    '"product": "...", "quantity_captured": true|false, "specs_captured": <int>, "qty_asks": <int, times the bot asked '
    'for quantity>, "spec_asks": <int, times the bot asked for any specification>, "seller_pitched": true|false, '
    '"params": {"<parameter>": {"verdict": "Non-fatal|Fatal", "reason": "<max 12 words>"}}, '
    '"overall": "Pass|Non-fatal|Fatal", "summary": "<one sentence>"}\n'
    f"Parameters: {', '.join(PARAMS)}. In params list ONLY parameters judged Non-fatal or Fatal; omit Pass and N/A."
)


def _json_chat(messages: list[dict], **kw):
    """as_json chat; a malformed JSON reply is retried once with light sampling. That is a different request,
    so it gets its own cache entry instead of replaying the same broken reply."""
    try:
        return sarvam.chat(messages, as_json=True, temperature=0.0, **kw)
    except ValueError:                   # json.JSONDecodeError
        return sarvam.chat(messages, as_json=True, temperature=0.3, **kw)


def judge(transcript_text: str) -> dict:
    """sarvam-105b with reasoning: on spot checks it judged outcomes more reliably than the fast model."""
    return _json_chat([{"role": "system", "content": JUDGE_SYSTEM},
                       {"role": "user", "content": f"TRANSCRIPT:\n{transcript_text}\n\n{JUDGE_SCHEMA}"}],
                      max_tokens=16000)   # reasoning model: needs headroom


PERSONA_PROMPT = (
    "From this real VANI call transcript, describe the BUYER so another model can role-play them in a new call. "
    'Reply with JSON only: {"product": "...", "category": "...", "language": "hindi|english|hinglish|telugu|other", '
    '"gender": "male|female|unknown", "quantity": "<what they said, or null>", "specs": ["..."], '
    '"buying_intent": "strong|weak|none", "patience": "high|medium|low", "behaviour": "<2 sentences: how they '
    'talk, interrupt, object, or hang up>", "quotes": ["<up to 3 short real buyer lines, verbatim>"]}'
)


def persona(transcript_text: str) -> dict:
    return _json_chat([{"role": "user", "content": f"{PERSONA_PROMPT}\n\nTRANSCRIPT:\n{transcript_text}"}],
                      model=sarvam.FAST_MODEL, max_tokens=3000)


def analyse(path: Path) -> dict:
    """One transcript file -> QA verdicts + persona + measured dead air."""
    tr = json.loads(path.read_text())
    ts = turns(tr)
    rec = {"recording": path.name.removesuffix(".json"), "language": tr.get("language_code"), "turns": ts,
           "duration_s": round(ts[-1]["end"], 1) if ts else 0.0, "dead_air": dead_air(ts),
           "usable": bool(ts) and buyer_said_something(ts)}
    if rec["usable"]:
        text = render(ts)
        rec["qa"] = judge(text)
        rec["persona"] = persona(text)
    return rec


def load_real_calls(path: Path | None = None) -> list[dict]:
    path = path or REAL_CALLS          # resolved at call time so tests can redirect it
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
