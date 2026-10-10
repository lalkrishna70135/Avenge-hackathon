"""Simulated VANI calls: a bot running prompt A or B talks to a buyer role-played from a real call's persona.

Bot prompt = condensed VANI core + this call's collection fields + the three variant-controlled sections verbatim.
  * Collection fields: the product's quantity units and spec options (abtest/catalog.py), which live VANI
    renders into its prompt template; without them the bot never asks for quantity or specifications.
  * A: all three sections as in the live prompt (byte-identical control for every experiment).
  * B for one experiment: the same text with only that experiment's section swapped for Sarvam's rewrite.
The core summary is built with those sections removed and told not to restate ask limits, follow-up
rules or greeting handling, so it cannot contradict a B section.

The judge (realcalls.judge) sees only the transcript: no variant label, no prompt text.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from . import catalog, realcalls, sarvam
from .pilot import AA_ID  # noqa: F401  (the A/A arm's experiment id)

ROOT = Path(__file__).resolve().parent.parent
BASE_PROMPT = ROOT / "data" / "base_prompt.txt"
CORE = sarvam.SARVAM_DIR / "vani_core.txt"
PILOT_DIR = ROOT / "runs" / "pilot"
CONVERSATIONS = PILOT_DIR / "conversations.jsonl"
MAX_BOT_TURNS = 14
WORDS_PER_SECOND = 2.5          # spoken Hinglish, for the duration estimate
SECONDS_PER_TURN = 1.0          # turn-taking gap
END_RE = re.compile(r"\[END_CALL(?:\s+disposition\s*=\s*([^\]]*))?\]", re.I)
HANGUP_RE = re.compile(r"\[HANGUP\]", re.I)


# ---------------------------------------------------------------------------- prompt assembly
AA_INFO = {"title": "A/A check: the control prompt against itself", "lines": None, "b_text": None,
           "diff": "No change: each persona talks to prompt A a second time, independently. Whatever lift this shows "
                   "is noise, the yardstick for the other changes' lifts."}


def experiments() -> dict[str, dict]:
    """experiment_id -> {lines: (lo, hi) 1-based inclusive, b_text, diff, title} for scenarios with a B prompt."""
    out = {}
    for path in sorted((ROOT / "scenarios").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        vb = cfg.get("variant_b") or {}
        pf = vb.get("prompt_file")
        if not pf or cfg.get("pilot_of"):
            continue
        text = (ROOT / pf).read_text()
        header = next(l for l in text.splitlines() if l.startswith("# Replaces lines"))
        lo, hi = (int(x) for x in header.split("lines ")[1].split(" of")[0].split("-"))
        body = "\n".join(l for l in text.splitlines() if not l.startswith("# ")).strip()
        out[cfg["experiment_id"]] = {"lines": (lo, hi), "b_text": body, "diff": vb.get("diff"), "title": cfg["title"]}
    return out


CORE_SYSTEM = (
    "You condense part of the system prompt of VANI, IndiaMART's buyer-side voice bot, into a compact operating "
    "guide another model will follow on live calls. Keep every rule that changes behaviour: role and persona, "
    "language rules, intent handling, objection handling, lead-quality rules, the End-Call Disposition Table "
    "(keep disposition values exactly), closing lines, TTS/speech style. Drop examples, repetition and template "
    "syntax. IMPORTANT: do NOT state any numeric ask limits, attempt counts, follow-up/re-ask rules for quantity, "
    "specifications or other fields, and do NOT describe how to answer a greeting. Where the text covers those, "
    "write only 'see COLLECTION AND GREETING RULES'. Reply with the condensed rules only, as short bullet points."
)


MERGE_NOTE = (" The input is already-condensed rules from consecutive parts of the prompt: merge them, remove "
              "duplicates and restatements, and keep it to about a third of the input length.")


def build_core(chunk_lines: int = 200, target_chars: int = 22000) -> str:
    """Summarise the base prompt (minus every variant-controlled section) chunk by chunk."""
    lines = BASE_PROMPT.read_text().splitlines()
    removed = set()
    for e in experiments().values():
        lo, hi = e["lines"]
        removed.update(range(lo - 1, hi))
    kept = [l for i, l in enumerate(lines) if i not in removed]
    parts = []
    for i in range(0, len(kept), chunk_lines):
        chunk = "\n".join(kept[i:i + chunk_lines])
        parts.append(sarvam.chat([{"role": "system", "content": CORE_SYSTEM},
                                  {"role": "user", "content": f"PROMPT PART {i // chunk_lines + 1}:\n{chunk}"}],
                                 temperature=0.0, max_tokens=16000))
    core = "\n\n".join(parts)
    # second pass: the first pass is ~60k chars, and at that length sarvam-105b-conversations returns
    # empty replies mid-call. Merge and de-duplicate down to roughly a third.
    while len(core) > target_chars:
        groups = [parts[i:i + 3] for i in range(0, len(parts), 3)]
        parts = [sarvam.chat([{"role": "system", "content": CORE_SYSTEM + MERGE_NOTE},
                              {"role": "user", "content": "\n\n".join(g)}], temperature=0.0, max_tokens=16000)
                 for g in groups]
        core = "\n\n".join(parts)
    sarvam.write_atomic(CORE, core)
    return core


SIM_CONTEXT = (
    "CALL CONTEXT (simulation): your name is Tania. The buyer called a seller about {product}; the seller was "
    "unavailable so the call reached you. A live alternate seller IS available for transfer (is_enrich = 1): "
    "confirm the product, collect details as the COLLECTION rules allow, then pitch the live seller. When the buyer "
    "agrees or declines, say the transfer or closing line and end the call.\n"
    "You have NO tools: wherever the prompt says to invoke a tool or set a variable, just say the spoken line. "
    "Never reply with nothing.\n"
    "OUTPUT PROTOCOL: reply with exactly what you say aloud on the phone, nothing else: one short turn, in the "
    "buyer's language (Hindi in Devanagari mixed with English words is fine). When the call ends, finish your last "
    "turn with [END_CALL disposition=<BL Enriched if you transferred the buyer to the live seller with the requirement "
    "captured; BL Generated if a lead was created without transfer; otherwise the buyer_disposition from the End-Call "
    "Disposition Table>]."
)


FIELDS_BLOCK = (
    "=== THIS CALL'S COLLECTION FIELDS (filled per call, as live VANI fills its prompt template) ===\n"
    "quantity_unit_options: {units}\n"
    "specification_options (spec name: common options, ask in this order): {specs}\n"
    "Active Collection Objectives for this call, top to bottom: {queue}. This list replaces any other "
    "active-queue list in the rules above."
)


def fields_block(fields: dict) -> str:
    """The template variables the live prompt is rendered with: quantity and specification collection are
    active only when the product has units / spec options (base_prompt.txt, Collection Objective Queue)."""
    units, specs = fields.get("quantity_unit_options") or [], fields.get("specification_options") or {}
    queue = (["Quantity Collection"] if units else []) + (["Specification Collection"] if specs else []) \
        + ["Buyer Name Collection", "Location Collection"]
    return FIELDS_BLOCK.format(units=json.dumps(units, ensure_ascii=False),
                               specs=json.dumps(specs, ensure_ascii=False) if specs else "none for this product",
                               queue=", ".join(queue))


def bot_system(arm: str, exp_id: str | None, product: str, core: str | None = None,
               fields: dict | None = None) -> str:
    """arm 'A' = live prompt sections; arm 'B' = exp_id's section rewritten, others unchanged.
    `fields` defaults to the product's IndiaMART category fields (catalog.collection_fields)."""
    core = core if core is not None else CORE.read_text()
    fields = fields if fields is not None else catalog.collection_fields(product)
    base = BASE_PROMPT.read_text().splitlines()
    sections = []
    for eid, e in experiments().items():
        lo, hi = e["lines"]
        text = e["b_text"] if (arm == "B" and eid == exp_id) else "\n".join(base[lo - 1:hi])
        sections.append(text)
    return (SIM_CONTEXT.format(product=product) + "\n\n=== VANI OPERATING RULES ===\n" + core
            + "\n\n" + fields_block(fields)
            + "\n\n=== COLLECTION AND GREETING RULES (follow exactly; they override the rules above) ===\n"
            + "\n\n".join(sections))


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------- buyer
BUYER_SYSTEM = (
    "You are role-playing a real IndiaMART buyer on a phone call. You called a seller about {product}; the seller "
    "didn't pick up and the call was redirected to IndiaMART's assistant. Stay in character:\n"
    "- Language: {language}. Speak like a phone caller: short, informal, sometimes incomplete.\n"
    "- Buying intent: {buying_intent}. Patience: {patience}. Behaviour: {behaviour}\n"
    "- What you know: quantity {quantity}; details {specs}. Share them only when asked, the way this buyer would.\n"
    "- Real lines this buyer said (match their style): {quotes}\n"
    "Most real buyers on these calls cooperate: they answer reasonable questions, sometimes briefly or partly, and "
    "accept being connected to a seller if they want to buy. Hang up only if the assistant asks for a detail again "
    "after you have already answered or refused it twice, or after you have clearly said twice that you are not "
    "interested; with low patience, once is enough. Never hang up in your first three replies.\n"
    "Reply with only what you say aloud. If you hang up, end your reply with [HANGUP]."
)


def buyer_system(p: dict) -> str:
    return BUYER_SYSTEM.format(product=p.get("product") or "a product", language=p.get("language") or "hinglish",
                               buying_intent=p.get("buying_intent") or "unclear", patience=p.get("patience") or "medium",
                               behaviour=p.get("behaviour") or "", quantity=p.get("quantity") or "not decided",
                               specs=", ".join(p.get("specs") or []) or "none decided",
                               quotes=" | ".join(p.get("quotes") or []))


# ---------------------------------------------------------------------------- one call
def clean(text: str) -> str:
    return HANGUP_RE.sub("", END_RE.sub("", text)).strip()


def bot_reply(system: str, history: list[dict], replicate: int = 0) -> str:
    """history: [{speaker, text}] so far. The bot sees the buyer as 'user'.
    sarvam-105b-conversations occasionally returns an empty reply mid-call (no tool call, no text);
    it happens where the prompt says to call a tool, so that turn is written by `action_line`."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "[call connected] Hello?"}]
    for t in history:
        msgs.append({"role": "assistant" if t["speaker"] == "bot" else "user", "content": t["raw"]})
    try:
        return sarvam.chat(msgs, model=sarvam.FAST_MODEL, temperature=0.5, max_tokens=600, replicate=replicate)
    except sarvam.EmptyReply:
        return action_line(history, replicate)


ACTION_SYSTEM = (
    "VANI, IndiaMART's voice assistant, just performed an action instead of speaking (almost always transferring "
    "the buyer to the live seller, or ending the call). Given the call so far, write the one short line VANI says "
    "aloud at this moment, in the buyer's language, and finish with [END_CALL disposition=<BL Enriched if the "
    "buyer is being connected to the seller with the requirement captured; BL Generated if a lead is created "
    "without transfer; otherwise No Product Requirement>]. Reply with that line only."
)


def action_line(history: list[dict], replicate: int = 0) -> str:
    """Fallback when the bot model returns nothing: it does so where the full prompt says to invoke a tool
    (seller transfer / end_interaction), which this simulation has no API for."""
    convo_text = realcalls.render([{"speaker": t["speaker"], "text": t["text"]} for t in history])
    line = sarvam.chat([{"role": "system", "content": ACTION_SYSTEM}, {"role": "user", "content": convo_text}],
                       model=sarvam.FAST_MODEL, temperature=0.3, max_tokens=300, replicate=replicate)
    return line if END_RE.search(line) else f"{line} [END_CALL disposition=Other]"


def buyer_reply(system: str, history: list[dict], replicate: int = 0) -> str:
    msgs = [{"role": "system", "content": system}]
    for t in history:
        msgs.append({"role": "user" if t["speaker"] == "bot" else "assistant", "content": clean(t["raw"])})
    return sarvam.chat(msgs, model=sarvam.FAST_MODEL, temperature=0.8, max_tokens=300, replicate=replicate)


def simulate(persona: dict, arm: str, exp_id: str, persona_id: str, core: str | None = None,
             fields: dict | None = None, replicate: int = 0) -> dict:
    """One simulated call. `fields` pins the collection fields so a persona's A and B calls get identical ones.
    `replicate` > 0 gives an independent call with the same prompts (the A/A check), not a replay of the cache."""
    product = persona.get("product") or "a product"
    fields = fields if fields is not None else catalog.collection_fields(product)
    system = bot_system(arm, exp_id, product, core, fields)
    bsys = buyer_system(persona)
    history, ended, disposition = [], "turn_limit", None
    for _ in range(MAX_BOT_TURNS):
        raw = bot_reply(system, history, replicate)
        history.append({"speaker": "bot", "raw": raw, "text": clean(raw)})
        m = END_RE.search(raw)
        if m:
            ended, disposition = "bot_ended", (m.group(1) or "").strip().strip("'\"") or None
            break
        raw = buyer_reply(bsys, history, replicate)
        history.append({"speaker": "buyer", "raw": raw, "text": clean(raw)})
        if HANGUP_RE.search(raw):
            ended = "buyer_hung_up"
            break
    turns = [{"speaker": t["speaker"], "text": t["text"]} for t in history if t["text"]]
    words = sum(len(t["text"].split()) for t in turns)
    transcript = realcalls.render(turns) + ("\n[buyer hung up]" if ended == "buyer_hung_up" else "")
    return {"persona_id": persona_id, "experiment_id": exp_id, "arm": arm, "prompt_sha": prompt_sha(system),
            "fields": fields, "turns": turns, "ended": ended, "bot_disposition": disposition,
            "est_duration_s": round(words / WORDS_PER_SECOND + SECONDS_PER_TURN * len(turns), 1),
            "judge": realcalls.judge(transcript)}


def load_conversations(path: Path | None = None) -> list[dict]:
    path = path or CONVERSATIONS          # resolved at call time so tests can redirect it
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
