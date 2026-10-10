"""Prompt-lab dashboard tabs (Real calls, Prompt lab, Live call) on small fixture data, with Sarvam mocked."""
import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from abtest import convo, realcalls, sarvam, store, voice
from abtest.loader import ROOT

from test_prompt_lab import _wav

APP = str(ROOT / "app" / "dashboard.py")
JUDGE = {"lead_created": True, "disposition": "BL Generated", "overall": "Non-fatal", "qty_asks": 1, "spec_asks": 1,
         "seller_pitched": True, "params": {"Looping Behavior": {"verdict": "Non-fatal", "reason": "asked twice"}},
         "summary": "Lead captured."}


@pytest.fixture
def lab(db_path, tmp_path, monkeypatch):
    turns = [{"speaker": "bot", "text": "नमस्ते, TMT bars की requirement है?", "start": 0, "end": 3},
             {"speaker": "buyer", "text": "हाँ 200 kg चाहिए", "start": 4, "end": 6}]
    real = [{"recording": f"r{i}.mp3", "language": "hi-IN", "turns": turns, "duration_s": 6.0,
             "dead_air": {"gaps_over_5s": 0, "longest_gap_s": 0.0}, "usable": True,
             "qa": {**JUDGE, "lead_created": i % 2 == 0}, "persona": {"product": "TMT bars", "gender": "male"}}
            for i in range(12)]
    (tmp_path / "real_calls.jsonl").write_text("\n".join(json.dumps(r) for r in real))
    convs = []
    for i in range(12):
        for arm, eid in (("A", "control"), ("B", "exp-win")):
            convs.append({"persona_id": f"r{i}.mp3", "experiment_id": eid, "arm": arm, "prompt_sha": "x",
                          "turns": turns, "ended": "bot_ended", "bot_disposition": "BL Generated",
                          "est_duration_s": 30.0, "judge": {**JUDGE, "lead_created": arm == "B" or i < 3}})
    convs += [{**convs[1], "persona_id": "r0.mp3", "experiment_id": "exp-harm"}]      # 1 pair only: examples
    (tmp_path / "conversations.jsonl").write_text("\n".join(json.dumps(c) for c in convs))
    (tmp_path / "core.txt").write_text("CORE RULES")
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "r0.mp3__A__exp-win.wav").write_bytes(_wav())
    monkeypatch.setattr(store, "DB_PATH", db_path)
    monkeypatch.setattr(realcalls, "REAL_CALLS", tmp_path / "real_calls.jsonl")
    monkeypatch.setattr(realcalls, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "u1.mp3.json").write_text(json.dumps({"language_code": "hi-IN", "diarized_transcript": {
        "entries": [{"speaker_id": "0", "transcript": "India Mart से, requirement है?", "start_time_seconds": 0,
                     "end_time_seconds": 2}, {"speaker_id": "1", "transcript": "हाँ 50 piece चाहिए Delhi में",
                     "start_time_seconds": 3, "end_time_seconds": 5}]}}))
    monkeypatch.setattr(convo, "CONVERSATIONS", tmp_path / "conversations.jsonl")
    monkeypatch.setattr(convo, "PILOT_DIR", tmp_path)
    monkeypatch.setattr(convo, "CORE", tmp_path / "core.txt")
    monkeypatch.setattr(convo.catalog, "collection_fields",       # no catalogue files or Sarvam units call
                        lambda product: {"quantity_unit_options": ["Ton"], "specification_options": {}})
    monkeypatch.setattr(voice, "AUDIO_DIR", tmp_path / "audio")
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    return at


def test_real_calls_and_prompt_lab_tabs(lab):
    lab.selectbox(key="lab_exp").set_value("exp-win").run()
    labels = {m.label: m.value for m in lab.metric}
    assert labels["Recordings transcribed"] == "1" and labels["Lead created (judge)"] == "50%"
    assert labels["Persona pairs"] == "12" and labels["Lead rate A → B"] == "25% → 100%"
    for opt in ("Lead created", "No lead", "Overall Fatal", "Not yet audited"):
        lab.selectbox(key="rc_filter").set_value(opt).run()
        assert not lab.exception
    assert lab.selectbox(key="rc_call").label == "Call (1)"            # the unaudited transcript
    lab.selectbox(key="lab_exp").set_value("exp-harm").run()          # too few pairs: examples, no estimate
    assert not lab.exception and any("Only 1 simulated A/B pair" in w.value for w in lab.warning)


def test_live_call_routes_talks_and_audits(lab, monkeypatch):
    calls = []

    def fake_chat(messages, **kw):
        if "audit calls" in messages[0]["content"]:
            return JUDGE
        calls.append(messages)
        return "नमस्ते! आपको कितनी quantity चाहिए?" if len(calls) == 1 else \
            "धन्यवाद, seller आपको call करेंगे। [END_CALL disposition=BL Generated]"

    monkeypatch.setattr(sarvam, "chat", fake_chat)
    monkeypatch.setattr(voice, "synth", lambda text, speaker, language: [_wav()])
    lab.text_input(key="live_caller").set_value("12345")
    lab.button(key="live_start").click().run()
    assert not lab.exception and len(calls) == 1
    assert "CORE RULES" in calls[0][0]["content"]                     # bot runs on the condensed prompt
    lab.chat_input(key="live_text").set_value("200 kg chahiye").run()
    assert not lab.exception and len(calls) == 2
    assert any("The bot ended the call." in s.value for s in lab.success)
    lab.button(key="live_end").click().run()
    assert any(m.label == "Overall" and m.value == "Non-fatal" for m in lab.metric)
    lab.button(key="live_new").click().run()
    assert not lab.exception and lab.button(key="live_start")
