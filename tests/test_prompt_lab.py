"""Offline tests for the Sarvam prompt lab (no network: Sarvam calls are mocked)."""
import io
import wave

import numpy as np
import pytest

from abtest import convo, pilot, realcalls, sarvam, voice


def test_parse_json_handles_fences_and_prose():
    assert sarvam.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert sarvam.parse_json('Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    with pytest.raises(Exception):
        sarvam.parse_json("no json here")
    assert sarvam.parse_json('{\n "product": null",\n "specs": [],\n "q": null"\n}') == \
        {"product": None, "specs": [], "q": None}                      # stray quote after null
    assert sarvam.parse_json('{"quote": "said null\\" here"}') == {"quote": 'said null" here'}   # text untouched
    assert sarvam.parse_json('{"a": "was null", "b": 1}') == {"a": "was null", "b": 1}   # valid JSON never repaired


def test_cache_key_covers_every_request_field():
    base = dict(model="m", messages=[{"role": "user", "content": "x"}], temperature=0.2, max_tokens=10,
                reasoning_effort="low")
    keys = {sarvam.cache_key(**base)}
    for k, v in [("temperature", 0.3), ("max_tokens", 11), ("model", "n"), ("reasoning_effort", "high")]:
        keys.add(sarvam.cache_key(**{**base, k: v}))
    assert len(keys) == 5


def test_end_and_hangup_markers():
    raw = "Dhanyavaad, seller aapko call karenge. [END_CALL disposition=BL Generated]"
    assert convo.END_RE.search(raw).group(1).strip() == "BL Generated"
    assert convo.clean(raw) == "Dhanyavaad, seller aapko call karenge."
    assert convo.clean("Nahi chahiye [HANGUP]") == "Nahi chahiye"


def test_bot_speaker_dead_air_and_usable():
    entries = [{"speaker_id": "2", "transcript": "Hello", "start_time_seconds": 0, "end_time_seconds": 1},
               {"speaker_id": "1", "transcript": "नमस्ते India Mart से बोल रही हूँ, requirement है ना",
                "start_time_seconds": 1.5, "end_time_seconds": 5},
               {"speaker_id": "2", "transcript": "हाँ 200 kg चाहिए Pune में", "start_time_seconds": 11,
                "end_time_seconds": 13}]
    assert realcalls.bot_speaker(entries) == "1"
    ts = realcalls.turns({"diarized_transcript": {"entries": entries}})
    assert [t["speaker"] for t in ts] == ["buyer", "bot", "buyer"]
    assert realcalls.dead_air(ts) == {"gaps_over_5s": 1, "longest_gap_s": 6.0}
    assert realcalls.buyer_said_something(ts)
    assert not realcalls.buyer_said_something([{"speaker": "buyer", "text": "Hello hello"}])


REAL_COLLECTION_FIELDS = convo.catalog.collection_fields      # captured before the autouse fixture patches it
FIELDS = {"mcat": "TMT Bars", "quantity_unit_options": ["Ton", "Kg"],
          "specification_options": {"Diameter": ["8 mm", "12 mm"], "Grade": ["Fe 500", "Fe 550"]}}


@pytest.fixture(autouse=True)
def _fixed_fields(monkeypatch):
    """Prompt assembly must not depend on the catalogue files or on Sarvam (units)."""
    monkeypatch.setattr(convo.catalog, "collection_fields", lambda product: FIELDS)


def test_fields_block_renders_the_template_queue():
    block = convo.fields_block(FIELDS)
    assert '"Ton", "Kg"' in block and '"Diameter"' in block
    assert "Quantity Collection, Specification Collection, Buyer Name Collection, Location Collection" in block
    bare = convo.fields_block({"quantity_unit_options": [], "specification_options": {}})
    assert "Quantity Collection" not in bare and "Specification Collection" not in bare
    assert convo.fields_block(FIELDS) in convo.bot_system("A", None, "TMT bars", core="CORE")


def test_catalog_match_prefers_full_and_rare_token_matches(monkeypatch):
    from abtest import catalog
    index = {"1": {"name": "Portable Mini Printer", "pop": 900, "specs": [["Print", ["A4"]]]},
             "2": {"name": "Portable Ultrasound Machine", "pop": 10, "specs": [["Probe", ["Convex"]]]},
             "3": {"name": "Ultrasound Gel", "pop": 50, "specs": [["Pack", ["250 ml"]]]},
             "4": {"name": "Pressure Cooker", "pop": 500, "specs": [["Capacity", ["5 L"]]]},
             "5": {"name": "Hawkins Pressure Cooker", "pop": 40, "specs": [["Capacity", ["3 L"]]]}}
    for i in range(6, 40):          # padding so idf separates common from rare tokens
        index[str(i)] = {"name": f"Mini Portable Machine {i}", "pop": 1, "specs": [["X", ["y"]]]}
    monkeypatch.setattr(catalog, "_index", lambda: _build(index))
    assert catalog.match("Hawkins Contoura Black pressure cookers")["mcat"] == "Hawkins Pressure Cooker"
    assert catalog.match("Mini Portable Ultrasound")["mcat"] == "Portable Ultrasound Machine"
    assert catalog.match("gold necklace") is None


def test_collection_fields_computed_once_per_product_across_threads(monkeypatch):
    """Parallel conversations for one persona must get identical fields, even though Sarvam can name
    different units on every call."""
    import itertools
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from abtest import catalog
    answers, lock = itertools.cycle([["Piece", "Set"], ["Piece", "Carton"], ["Ton"]]), threading.Lock()

    def fake_chat(messages, **kw):
        time.sleep(0.05)                     # widen the window a race would need
        with lock:
            return next(answers)
    monkeypatch.setattr(catalog.sarvam, "chat", fake_chat)
    monkeypatch.setattr(catalog, "match", lambda product: None)
    catalog._fields.cache_clear()
    try:
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda _: REAL_COLLECTION_FIELDS("massage bed"), range(8)))
    finally:
        catalog._fields.cache_clear()
    assert all(g == got[0] for g in got) and got[0]["quantity_unit_options"] == ["Piece", "Set"]


def _build(index):
    import math
    from abtest import catalog
    inverted = {}
    for mid, m in index.items():
        m["tokens"] = catalog.tokens(m["name"])
        for t in m["tokens"]:
            inverted.setdefault(t, []).append(mid)
    idf = {t: math.log(len(index) / len(ms)) for t, ms in inverted.items()}
    return index, inverted, idf


def test_bot_prompts_control_identical_and_b_swaps_only_its_section():
    exps = convo.experiments()
    assert set(exps) == {"exp-win", "exp-harm", "exp-null"}
    a = convo.bot_system("A", None, "TMT bars", core="CORE")
    assert all(convo.bot_system("A", e, "TMT bars", core="CORE") == a for e in exps)
    base = convo.BASE_PROMPT.read_text().splitlines()
    for eid, e in exps.items():
        b = convo.bot_system("B", eid, "TMT bars", core="CORE")
        lo, hi = e["lines"]
        assert b == a.replace("\n".join(base[lo - 1:hi]), e["b_text"])
        assert b != a


def test_paired_estimate_and_effect_translation():
    def c(pid, arm, exp, lead, dur):
        return {"persona_id": pid, "arm": arm, "experiment_id": exp, "est_duration_s": dur, "ended": "bot_ended",
                "turns": [{}] * 4, "bot_disposition": "BL Generated" if lead else "BL Deleted",
                "judge": {"lead_created": lead, "overall": "Pass", "qty_asks": 1, "spec_asks": 1}}
    convs = [c(f"p{i}", "A", "control", i < 5, 60) for i in range(10)]
    convs += [c(f"p{i}", "B", "exp-win", i < 8, 66) for i in range(10)]
    s = pilot.paired(convs, "exp-win")
    assert s["pairs"] == 10 and s["lift_pp"] == pytest.approx(0.3)
    assert s["discordant"] == {"B_better": 3, "A_better": 0} and 0 < s["p_value"] < 0.3
    assert s["duration_ratio"] == pytest.approx(1.1) and s["A"]["bot_judge_agree"] == 1.0
    eff = pilot.to_true_effect(s, p_connected=0.7)        # odds (5.5/5.5 -> 8.5/2.5): OR 3.4 on odds 7/3
    assert eff["odds_ratio"] == pytest.approx(3.4) and not eff["clamped"]
    assert eff["success_pp"] == pytest.approx(7 / 3 * 3.4 / (1 + 7 / 3 * 3.4) - 0.7, abs=1e-4)
    extreme = {**s, "A": {**s["A"], "lead_rate": 0.0}, "B": {**s["B"], "lead_rate": 1.0}}
    eff = pilot.to_true_effect(extreme, p_connected=0.7)  # 0% -> 100%: finite OR, then the safeguard clamp
    assert eff["clamped"] and eff["success_pp"] == pytest.approx(0.95 * 0.3)
    assert pilot.paired(convs, "exp-harm") is None


def _wav(rate=22050, n=2205, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * n)
    return buf.getvalue()


def test_wav_join_and_text_split():
    joined = voice.join_wavs([_wav(), _wav()], gap_s=0.5)
    assert voice.wav_seconds(joined) == pytest.approx(0.1 + 0.5 + 0.1)
    with pytest.raises(ValueError):
        voice.join_wavs([_wav(), _wav(rate=16000)])
    long = "यह एक वाक्य है। " * 60
    parts = voice.split_text(long, limit=100)
    assert all(len(p) <= 100 for p in parts) and "".join(parts).replace(" ", "") == long.replace(" ", "")
    assert voice.language_for("नमस्ते") == "hi-IN" and voice.language_for("Hello sir") == "en-IN"


def test_simulate_with_mocked_sarvam(monkeypatch):
    seen = {}

    def fake_chat(messages, **kw):
        system = messages[0]["content"]
        if "audit calls" in system:                                   # judge
            seen["judge_input"] = messages[1]["content"]
            return {"lead_created": True, "overall": "Pass", "qty_asks": 1, "spec_asks": 0}
        if "role-playing a real IndiaMART buyer" in system:           # buyer
            return "200 kg chahiye"
        n_bot = sum(m["role"] == "assistant" for m in messages)       # bot
        return "Kitni quantity chahiye?" if n_bot == 0 else "Theek hai, seller call karenge. [END_CALL disposition=BL Generated]"

    monkeypatch.setattr(sarvam, "chat", fake_chat)
    out = convo.simulate({"product": "TMT bars", "gender": "male"}, "B", "exp-win", "p1", core="CORE")
    assert out["ended"] == "bot_ended" and out["bot_disposition"] == "BL Generated"
    assert [t["speaker"] for t in out["turns"]] == ["bot", "buyer", "bot"]
    assert "END_CALL" not in seen["judge_input"]
    assert "exp-win" not in seen["judge_input"] and "variant" not in seen["judge_input"].lower()   # judge is blind
    assert out["est_duration_s"] > 0 and np.isfinite(out["est_duration_s"])


def test_malformed_json_is_retried_once_with_sampling(monkeypatch):
    seen = []

    def fake_chat(messages, as_json=False, temperature=0.0, **kw):
        seen.append(temperature)
        if temperature == 0.0:
            raise ValueError("Expecting ',' delimiter")
        return {"product": "coal"}
    monkeypatch.setattr(realcalls.sarvam, "chat", fake_chat)
    assert realcalls.persona("BOT: hi\nBUYER: coal chahiye") == {"product": "coal"}
    assert seen == [0.0, 0.3]


def test_replicate_is_an_independent_cached_sample(monkeypatch, tmp_path):
    """The A/A arm reuses prompt A, so without `replicate` it would replay A's cached conversation."""
    from types import SimpleNamespace
    calls = []

    def completions(**kw):
        calls.append(kw)
        msg = SimpleNamespace(content=f"reply {len(calls)}")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    monkeypatch.setattr(sarvam, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sarvam, "client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    msgs = [{"role": "user", "content": "Hello?"}]
    first = sarvam.chat(msgs)
    assert sarvam.chat(msgs) == first and len(calls) == 1               # cached replay
    second = sarvam.chat(msgs, replicate=1)
    assert second != first and len(calls) == 2                          # a fresh sample...
    assert "replicate" not in calls[-1]                                  # ...without sending the tag to the API
    assert sarvam.chat(msgs, replicate=1) == second and len(calls) == 2  # and itself cached


def test_aa_replicate_is_the_noise_yardstick_and_pools_into_the_control():
    def c(pid, arm, exp, lead):
        return {"persona_id": pid, "arm": arm, "experiment_id": exp, "est_duration_s": 60, "ended": "bot_ended",
                "turns": [{}] * 4, "bot_disposition": "", "judge": {"lead_created": lead, "overall": "Pass"}}
    convs = [c(f"p{i}", "A", "control", i < 2, ) for i in range(10)]           # first A call: 20% leads
    convs += [c(f"p{i}", "B", pilot.AA_ID, i < 6) for i in range(10)]          # replicate of A: 60% leads
    convs += [c(f"p{i}", "B", "exp-win", i < 5) for i in range(10)]            # the change: 50% leads
    aa, win = pilot.paired(convs, pilot.AA_ID), pilot.paired(convs, "exp-win")
    assert aa["lift_pp"] == pytest.approx(0.4) and aa["A"]["calls"] == 10       # A/A: replicate vs first A only
    assert win["A"]["calls"] == 20 and win["A"]["lead_rate"] == pytest.approx(0.4)   # pooled control
    assert win["lift_pp"] == pytest.approx(0.1)
