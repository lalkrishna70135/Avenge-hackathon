"""Sarvam prompt-lab tabs for the dashboard: Real calls, Prompt lab, Live call.

Everything except the Live call tab reads cached pipeline outputs, so it works offline. The Live
call tab creates the Sarvam client only when a call starts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from abtest import convo, realcalls, voice
from abtest.assigner import BUCKETS, bucket
from abtest.realcalls import PARAMS

ROOT = Path(__file__).resolve().parent.parent
VERDICT_COLOUR = {"Fatal": "#C0392B", "Non-fatal": "#E07A1F"}


@st.cache_data
def real_calls() -> list[dict]:
    return realcalls.load_real_calls()


@st.cache_data
def unaudited_calls() -> list[dict]:
    """Transcribed recordings that have not been through the (paid) quality audit yet: still worth listening to."""
    done = {r["recording"] for r in realcalls.load_real_calls()}
    out = []
    for p in sorted(realcalls.TRANSCRIPT_DIR.glob("*.json")):
        rec = p.name.removesuffix(".json")
        if rec in done:
            continue
        tr = json.loads(p.read_text())
        ts = realcalls.turns(tr)
        if ts and realcalls.buyer_said_something(ts):
            out.append({"recording": rec, "language": tr.get("language_code"), "turns": ts,
                        "duration_s": round(ts[-1]["end"], 1), "dead_air": realcalls.dead_air(ts), "qa": None})
    return out


@st.cache_data
def conversations() -> list[dict]:
    return convo.load_conversations()


def show_turns(turns: list[dict], height: int = 360):
    with st.container(height=height):
        for t in turns:
            with st.chat_message("assistant" if t["speaker"] == "bot" else "user",
                                 avatar="🤖" if t["speaker"] == "bot" else "🧑"):
                st.markdown(t["text"])


def scorecard(j: dict):
    if not j:
        st.caption("No judge result.")
        return
    c = st.columns(3)
    c[0].metric("Lead created", "yes" if j.get("lead_created") else "no", j.get("disposition"), delta_color="off")
    c[1].metric("Overall", j.get("overall", "–"))
    c[2].metric("Asks: qty / spec", f"{j.get('qty_asks', '–')} / {j.get('spec_asks', '–')}",
                "seller pitched" if j.get("seller_pitched") else "seller not pitched", delta_color="off")
    flags = [{"parameter": k, "verdict": v.get("verdict"), "reason": v.get("reason")}
             for k, v in (j.get("params") or {}).items() if v.get("verdict") in VERDICT_COLOUR]
    if flags:
        st.dataframe(pd.DataFrame(flags), hide_index=True, width="stretch")
    else:
        st.caption("No parameter flagged.")
    if j.get("summary"):
        st.caption(j["summary"])


# ---------------------------------------------------------------------------- real calls
def real_calls_tab(call_log_dispositions: pd.Series):
    calls = real_calls()
    if not calls:
        st.info("No analysed recordings yet. Run `scripts/transcribe_calls.py` then `scripts/analyse_calls.py`.")
        return
    usable = [c for c in calls if c.get("usable") and c.get("qa")]
    st.caption("Real VANI call recordings from the hackathon data, transcribed with Sarvam batch speech-to-text "
               "(saaras:v3, speaker diarization) and audited by sarvam-105b against the VANI Call Quality Matrix. "
               "The recordings don't map to call-log rows, so they are analysed on their own.")
    leads = [c for c in usable if c["qa"].get("lead_created")]
    c = st.columns(5)
    c[0].metric("Recordings transcribed", f"{len(list(realcalls.TRANSCRIPT_DIR.glob('*.json'))) or len(calls):,}")
    c[1].metric("Audited (Call Quality Matrix)", f"{len(usable):,}",
                f"{len(unaudited_calls()):,} more transcribed, not audited", delta_color="off")
    c[2].metric("Lead created (judge)", f"{len(leads) / max(len(usable), 1):.0%}")
    c[3].metric("Overall Fatal", f"{sum(u['qa'].get('overall') == 'Fatal' for u in usable) / max(len(usable), 1):.0%}")
    c[4].metric("Calls with dead air > 5 s", f"{sum(u['dead_air']['gaps_over_5s'] > 0 for u in usable):,}",
                "measured from diarization timestamps", delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        rows = []
        for p in PARAMS:
            vs = [(u["qa"].get("params") or {}).get(p, {}).get("verdict") for u in usable]
            rows.append({"parameter": p, "Fatal": vs.count("Fatal") / len(usable),
                         "Non-fatal": vs.count("Non-fatal") / len(usable)})
        df = pd.DataFrame(rows).sort_values("Fatal")
        fig = go.Figure()
        for v, colour in VERDICT_COLOUR.items():
            fig.add_bar(y=df["parameter"], x=100 * df[v], orientation="h", name=v, marker_color=colour)
        fig.update_layout(title="Where real calls fail the Call Quality Matrix", barmode="stack", height=480,
                          xaxis_title="% of calls", legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        judge_mix = pd.Series([u["qa"].get("disposition") or "Other" for u in usable]).value_counts(normalize=True)
        log_mix = call_log_dispositions.value_counts(normalize=True).head(8)
        fig = go.Figure()
        fig.add_bar(y=judge_mix.index, x=100 * judge_mix.values, orientation="h", name="recordings (judge)",
                    marker_color="#14213D")
        fig.add_bar(y=log_mix.index, x=100 * log_mix.values, orientation="h", name="call log (system)",
                    marker_color="#8A94A6")
        fig.update_layout(title="Outcome mix: recordings vs call log", barmode="group", height=480,
                          xaxis_title="% of calls", legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Listen to a real call")
    pending = unaudited_calls()
    f = st.columns([1, 1, 2])
    only = f[0].selectbox("Show", ["All", "Lead created", "No lead", "Overall Fatal", "Not yet audited"],
                          key="rc_filter")
    flag = f[1].selectbox("Flagged parameter", ["Any"] + PARAMS, key="rc_param")
    def keep(u):
        qa = u["qa"]
        if only == "Not yet audited":
            return qa is None
        if qa is None:
            return False
        ok = {"All": True, "Lead created": bool(qa.get("lead_created")), "No lead": not qa.get("lead_created"),
              "Overall Fatal": qa.get("overall") == "Fatal"}[only]
        return ok and (flag == "Any" or (qa.get("params") or {}).get(flag, {}).get("verdict") in VERDICT_COLOUR)

    pick = [u for u in usable + pending if keep(u)]
    if not pick:
        st.caption("No calls match.")
        return
    labels = {u["recording"]: (f"{(u.get('persona') or {}).get('product') or '?'} · {u['qa'].get('disposition')} · "
                               f"{u['qa'].get('overall')} · {u['duration_s']:.0f} s" if u["qa"] else
                               f"{u['recording']} · {u['duration_s']:.0f} s · not audited") for u in pick}
    rec = f[2].selectbox(f"Call ({len(pick)})", list(labels), format_func=labels.get, key="rc_call")
    u = next(x for x in pick if x["recording"] == rec)
    a, b = st.columns([3, 2])
    with a:
        audio = realcalls.REC_DIR / rec
        if audio.exists():
            st.audio(str(audio))
        st.caption(f"Language {u.get('language')} · dead air: {u['dead_air']['gaps_over_5s']} gaps over 5 s "
                   f"(longest {u['dead_air']['longest_gap_s']} s)")
        show_turns(u["turns"])
    with b:
        st.markdown("**Quality audit**")
        if u["qa"]:
            scorecard(u["qa"])
        else:
            st.caption("Transcribed but not audited: the audit run was stopped to save API credits.")
        with st.expander("Buyer persona extracted for simulation"):
            st.json(u.get("persona") or {})


# ---------------------------------------------------------------------------- prompt lab
MIN_PAIRS = 10          # below this the pilot is shown as example conversations, not an estimate


def prompt_lab_tab():
    from abtest import pilot
    convs = conversations()
    if not convs:
        st.info("No simulated calls yet. Run `scripts/build_core_prompt.py` then `scripts/simulate_pilot.py`.")
        return
    exps = {**convo.experiments(), convo.AA_ID: convo.AA_INFO}
    st.caption("Each prompt change can be tried in simulated calls before it gets live traffic. Two Sarvam agents "
               "talk: the VANI bot runs prompt A or B, and the buyer is role-played from a real recorded call's "
               "persona. Like live VANI, the bot gets the product's quantity units and spec options, here from "
               "IndiaMART's category data. A blind judge (sarvam-105b, Call Quality Matrix) scores each transcript "
               "without knowing the variant.")
    summary = {e: s for e in exps if (s := pilot.paired(convs, e))}
    if not summary:
        st.info("No A/B pairs yet.")
        return
    exp_id = st.selectbox("Prompt change", list(summary), key="lab_exp",
                          format_func=lambda e: f"{e}: {exps.get(e, {}).get('title', '')} ({summary[e]['pairs']} pairs)")
    s = summary[exp_id]
    st.markdown(f"_Change:_ {exps.get(exp_id, {}).get('diff', '')}")
    aa = summary.get(convo.AA_ID)
    if aa and exp_id != convo.AA_ID:
        st.caption(f"Noise yardstick: the A/A check (prompt A against an independent second call on prompt A) shows "
                   f"{100 * aa['lift_pp']:+.0f} pp [{100 * aa['lift_ci'][0]:+.0f}, {100 * aa['lift_ci'][1]:+.0f}] with "
                   f"no change at all. All changes share the same A calls, so their lifts move together.")
    if s["pairs"] < MIN_PAIRS:
        st.warning(f"Only {s['pairs']} simulated A/B pair(s) for this change (the run was stopped to save API credits), "
                   f"so these are example conversations, not an effect estimate. `scripts/simulate_pilot.py "
                   f"--personas 60` produces the estimate; it needs at least {MIN_PAIRS} pairs.")
    else:
        c = st.columns(6)
        c[0].metric("Persona pairs", s["pairs"])
        c[1].metric("Lead rate A → B", f"{s['A']['lead_rate']:.0%} → {s['B']['lead_rate']:.0%}",
                    f"{100 * s['lift_pp']:+.0f} pp [{100 * s['lift_ci'][0]:+.0f}, {100 * s['lift_ci'][1]:+.0f}]",
                    delta_color="off")
        c[2].metric("Paired test p", f"{s['p_value']:.3f}",
                    f"B better for {s['discordant']['B_better']} personas, A for {s['discordant']['A_better']}",
                    delta_color="off")
        c[3].metric("Qty + spec asks per call", f"{s['A']['mean_asks']:.1f} → {s['B']['mean_asks']:.1f}",
                    f"re-asking in {s['A'].get('reask_rate', 0):.0%} → {s['B'].get('reask_rate', 0):.0%} of calls",
                    delta_color="off", help="Did B's rule reach the bot? A re-ask is a field asked again after one "
                    "attempt (quantity twice, or specs more often than there are spec fields).")
        c[4].metric("Duration B/A (est.)", f"{s['duration_ratio']:.2f}",
                    f"[{s['duration_ratio_ci'][0]:.2f}, {s['duration_ratio_ci'][1]:.2f}]", delta_color="off")
        c[5].metric("Fatal · hang-up", f"{s['A']['fatal_rate']:.0%} → {s['B']['fatal_rate']:.0%}",
                    f"hang-ups {s['A']['hangup_rate']:.0%} → {s['B']['hangup_rate']:.0%}", delta_color="off")
        pilot_yaml = ROOT / "scenarios" / "pilot.yaml"
        if pilot_yaml.exists() and f"pilot_of: {exp_id}" in pilot_yaml.read_text():
            st.success("This pilot feeds the **exp-pilot** experiment: its estimated effect is the true effect the "
                       "A/B engine then has to detect at scale. See the Experiments tab.")

    a = {x["persona_id"]: x for x in convs if x["arm"] == "A"}
    b = {x["persona_id"]: x for x in convs if x["arm"] == "B" and x["experiment_id"] == exp_id}
    ids = sorted(set(a) & set(b))
    personas = {r["recording"]: r.get("persona") or {} for r in real_calls()}

    def audio_path(pid, arm):
        return voice.AUDIO_DIR / f"{pid}__{arm}__{exp_id}.wav"

    def label(pid):
        la, lb = (bool((x[pid].get("judge") or {}).get("lead_created")) for x in (a, b))
        voiced = " 🔊" if audio_path(pid, "A").exists() else ""
        return (f"{personas.get(pid, {}).get('product') or pid} · A {'lead' if la else 'no lead'} / "
                f"B {'lead' if lb else 'no lead'}{voiced}")

    ids.sort(key=lambda i: (not audio_path(i, "A").exists(), label(i)))
    pid = st.selectbox(f"Buyer persona ({len(ids)})", ids, format_func=label, key="lab_persona")
    if pid:
        with st.expander("Persona (from a real recorded call) and the collection fields the bot was given"):
            st.json(personas.get(pid, {}))
            if a[pid].get("fields"):
                st.markdown("**Collection fields for this call**: the product's IndiaMART category, its quantity "
                            "units and its two main spec fields, as live VANI fills its prompt template")
                st.json(a[pid]["fields"])
        cols = st.columns(2)
        for col, arm, conv in ((cols[0], "A", a[pid]), (cols[1], "B", b[pid])):
            with col:
                st.markdown(f"**{'A: current prompt' if arm == 'A' else 'B: rewritten prompt'}** · "
                            f"ended: {conv['ended'].replace('_', ' ')} · bot said: {conv.get('bot_disposition') or '–'} "
                            f"· ~{conv['est_duration_s']:.0f} s")
                if audio_path(pid, arm).exists():
                    st.audio(str(audio_path(pid, arm)))
                show_turns(conv["turns"])
                scorecard(conv.get("judge") or {})
    st.caption("Limits: buyers are simulated (seeded from real calls), so this pilot is directional; the bot runs on "
               "a condensed VANI prompt with the changed sections verbatim; every change shares the same control (two "
               "calls on prompt A per persona: the A call and the A/A replicate); duration is estimated from words "
               "spoken.")


# ---------------------------------------------------------------------------- live call
def _live_reset():
    for k in ("live_history", "live_last_audio", "live_reply_audio", "live_judge", "live_route"):
        st.session_state.pop(k, None)


def _speak(text: str):
    try:
        st.session_state["live_reply_audio"] = voice.join_wavs(voice.synth(text, voice.BOT_VOICE, voice.language_for(text)))
    except Exception as exc:          # noqa: BLE001
        st.session_state["live_reply_audio"] = None
        st.warning(f"Voice failed, showing text only: {type(exc).__name__}")


def _bot_turn():
    r = st.session_state["live_route"]
    raw = convo.bot_reply(r["system"], st.session_state["live_history"])
    st.session_state["live_history"].append({"speaker": "bot", "raw": raw, "text": convo.clean(raw)})
    _speak(convo.clean(raw))


def _buyer_turn(text: str):
    st.session_state["live_history"].append({"speaker": "buyer", "raw": text, "text": text})
    _bot_turn()


def live_call_tab(exps_by_id: pd.DataFrame):
    if not convo.CORE.exists():
        st.info("Build the bot prompt first: `scripts/build_core_prompt.py`.")
        return
    st.caption("Talk to the VANI bot yourself. Your caller ID goes through the experiment's hash router to pick "
               "prompt A or B, exactly like live traffic; you speak as the buyer (Sarvam speech-to-text), the bot "
               "answers with that prompt (sarvam-105b-conversations) and speaks (Sarvam bulbul:v3).")
    route = st.session_state.get("live_route")
    if route is None:
        lab = [e for e in exps_by_id.index if e in convo.experiments()]
        f = st.columns([2, 2, 2, 1])
        exp_id = f[0].selectbox("Experiment", lab, key="live_exp")
        cfg = json.loads(exps_by_id.loc[exp_id, "config_json"])
        caller = f[1].text_input("Your caller ID", value="98765", key="live_caller")
        product = f[2].text_input("Product you're calling about", value="TMT steel bars", key="live_product")
        bk = bucket(cfg["salt"], caller.strip())
        arm = "B" if bk < round(cfg["share_b"] * BUCKETS) else "A"
        override = f[3].selectbox("Variant", ["router", "A", "B"], key="live_override")
        arm = arm if override == "router" else override
        st.markdown(f"Router: sha256(`{cfg['salt']}:{caller.strip()}`) mod {BUCKETS} = **{bk}** → "
                    f"{'B' if bk < round(cfg['share_b'] * BUCKETS) else 'A'} at a {cfg['share_b']:.0%} split"
                    + (f" · overridden to **{arm}**" if override != "router" else f" · you get **{arm}**"))
        if st.button("📞 Start call", type="primary", key="live_start"):
            fields = convo.catalog.collection_fields(product)
            st.session_state["live_route"] = {"exp_id": exp_id, "arm": arm, "caller": caller, "product": product,
                                              "fields": fields,
                                              "system": convo.bot_system(arm, exp_id, product, fields=fields)}
            st.session_state["live_history"] = []
            try:
                with st.spinner("Connecting…"):
                    _bot_turn()
            except Exception as exc:  # noqa: BLE001
                _live_reset()
                st.error(f"Could not reach Sarvam: {type(exc).__name__}: {str(exc)[:200]}")
                return
            st.rerun()
        return

    st.markdown(f"**On call** · {route['exp_id']} · prompt **{route['arm']}** · caller `{route['caller']}` · "
                f"{route['product']}")
    fl = route.get("fields") or {}
    if fl:
        st.caption(f"IndiaMART category: {fl.get('mcat') or 'no match'} · units {', '.join(fl.get('quantity_unit_options') or [])}"
                   f" · specs {', '.join(fl.get('specification_options') or {}) or 'none'}")
    hist = st.session_state["live_history"]
    show_turns(hist, height=380)
    if st.session_state.get("live_reply_audio"):
        st.audio(st.session_state["live_reply_audio"], format="audio/wav", autoplay=True)
    ended = bool(hist and convo.END_RE.search(hist[-1]["raw"]))
    if not ended:
        mic = st.audio_input("Speak as the buyer", key=f"live_mic_{len(hist)}")
        if mic is not None:
            data = mic.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            if digest != st.session_state.get("live_last_audio"):      # reruns must not re-send the same clip
                st.session_state["live_last_audio"] = digest
                try:
                    with st.spinner("Listening…"):
                        from abtest import sarvam
                        r = sarvam.client().speech_to_text.transcribe(
                            file=("buyer.wav", data, "audio/wav"), model=realcalls.STT_MODEL, mode="codemix",
                            language_code="unknown")
                        text = (r.transcript or "").strip()
                    if text:
                        with st.spinner("Bot is answering…"):
                            _buyer_turn(text)
                        st.rerun()
                    st.warning("Didn't catch that; try again.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Speech-to-text failed: {type(exc).__name__}: {str(exc)[:200]}")
        typed = st.chat_input("…or type what the buyer says", key="live_text")
        if typed:
            try:
                with st.spinner("Bot is answering…"):
                    _buyer_turn(typed)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Bot failed: {type(exc).__name__}: {str(exc)[:200]}")
            st.rerun()
    else:
        st.success("The bot ended the call.")

    c = st.columns(2)
    if c[0].button("End call and audit it", key="live_end", disabled=len(hist) < 2):
        with st.spinner("Judging the call against the Call Quality Matrix…"):
            st.session_state["live_judge"] = realcalls.judge(realcalls.render(
                [{"speaker": t["speaker"], "text": t["text"]} for t in hist]))
    if c[1].button("New call", key="live_new"):
        _live_reset()
        st.rerun()
    if st.session_state.get("live_judge"):
        scorecard(st.session_state["live_judge"])
